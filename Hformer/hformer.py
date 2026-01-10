import torch
import torch.nn as nn
import torch.nn.functional as F
from .hformer_parts import *
from .hrnet_backbone import HRNetBackbone

class Hformer(nn.Module):
    """
    Hybrid CNN-Transformer for Fringe Projection Profilometry
    Input: 960x960 fringe image
    Output: 960x960 depth map

    Architecture combines:
    - HRNet backbone for multi-scale CNN features
    - Transformer encoder-decoder with skip connections
    """
    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.5):
        super().__init__()

        # Configuration for 960x960 input
        # Patch embedding size 4 gives 240x240 feature maps
        self.embed_dim = 32
        self.depths = [1, 1, 2, 2]  # Depth of each stage
        self.num_heads = [1, 1, 2, 4]  # Attention heads for each stage
        self.window_size = 8  # Window size for local attention
        self.mlp_ratio = 2.0
        self.drop_rate = 0.0
        self.attn_drop_rate = 0.0
        self.drop_path_rate = 0.4

        # CNN Backbone (HRNet-W18)
        # Outputs 4 feature maps with channels [18, 36, 72, 144]
        # For 960x960 input after stem (stride 4): 240x240
        self.backbone = HRNetBackbone(in_channels=in_channels)

        # Patch embeddings for each backbone output
        # b1: 240x240x18 -> 240x240x64
        # b2: 120x120x36 -> 120x120x128
        # b3: 60x60x72 -> 60x60x256
        # b4: 30x30x144 -> 30x30x512
        self.patch_embed1 = PatchEmbedding(18, self.embed_dim, patch_size=1)
        self.patch_embed2 = PatchEmbedding(36, self.embed_dim * 2, patch_size=1)
        self.patch_embed3 = PatchEmbedding(72, self.embed_dim * 4, patch_size=1)
        self.patch_embed4 = PatchEmbedding(144, self.embed_dim * 8, patch_size=1)

        # Fusion layers for encoder
        self.fusion2 = nn.Linear(self.embed_dim * 4, self.embed_dim * 2)  # For enc2_input
        self.fusion3 = nn.Linear(self.embed_dim * 8, self.embed_dim * 4)  # For enc3_input  
        self.fusion4 = nn.Linear(self.embed_dim * 16, self.embed_dim * 8) # For enc4_input

        # Stochastic depth decay rule
        dpr = [x.item() for x in torch.linspace(0, self.drop_path_rate, sum(self.depths))]

        # ENCODER LAYERS
        # Encoder 1: 240x240, 64 channels
        self.encoder1 = CATLayer(
            dim=self.embed_dim,
            depth=self.depths[0],
            num_heads=self.num_heads[0],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            drop_path=dpr[0:self.depths[0]],
            downsample=PatchProjection(self.embed_dim, self.embed_dim * 2)
        )

        # Encoder 2: 120x120, 128 channels
        self.encoder2 = CATLayer(
            dim=self.embed_dim * 2,
            depth=self.depths[1],
            num_heads=self.num_heads[1],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            drop_path=dpr[self.depths[0]:self.depths[0]+self.depths[1]],
            downsample=PatchProjection(self.embed_dim * 2, self.embed_dim * 4)
        )

        # Encoder 3: 60x60, 256 channels
        self.encoder3 = CATLayer(
            dim=self.embed_dim * 4,
            depth=self.depths[2],
            num_heads=self.num_heads[2],
            window_size=self.window_size,
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            drop_path=dpr[self.depths[0]+self.depths[1]:self.depths[0]+self.depths[1]+self.depths[2]],
            downsample=PatchProjection(self.embed_dim * 4, self.embed_dim * 8)
        )

        # BOTTLENECK: 30x30, 512 channels
        bottleneck_dpr = dpr[sum(self.depths[0:3]):sum(self.depths)]
        self.bottleneck = CATLayer(
            dim=self.embed_dim * 8,
            depth=self.depths[3],
            num_heads=self.num_heads[3],
            window_size=self.window_size // 2,  # Smaller window for smaller feature map
            mlp_ratio=self.mlp_ratio,
            qkv_bias=True,
            drop=self.drop_rate,
            attn_drop=self.attn_drop_rate,
            drop_path=bottleneck_dpr,
            downsample=None
        )

        # Dropout after bottleneck
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER LAYERS
        # Decoder 4: 30x30 -> 60x60, 512 -> 256
        self.decoder4 = nn.ModuleList([
            PatchExpand(self.embed_dim * 8, self.embed_dim * 4, scale=2),
            nn.Linear(self.embed_dim * 8, self.embed_dim * 4),  # Concat reduction
            CATLayer(
                dim=self.embed_dim * 4,
                depth=self.depths[2],
                num_heads=self.num_heads[2],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[self.depths[0]+self.depths[1]:self.depths[0]+self.depths[1]+self.depths[2]],
                downsample=None
            )
        ])

        # Decoder 3: 60x60 -> 120x120, 256 -> 128
        self.decoder3 = nn.ModuleList([
            PatchExpand(self.embed_dim * 4, self.embed_dim * 2, scale=2),
            nn.Linear(self.embed_dim * 4, self.embed_dim * 2),
            CATLayer(
                dim=self.embed_dim * 2,
                depth=self.depths[1],
                num_heads=self.num_heads[1],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[self.depths[0]:self.depths[0]+self.depths[1]],
                downsample=None
            )
        ])

        # Decoder 2: 120x120 -> 240x240, 128 -> 64
        self.decoder2 = nn.ModuleList([
            PatchExpand(self.embed_dim * 2, self.embed_dim, scale=2),
            nn.Linear(self.embed_dim * 2, self.embed_dim),
            CATLayer(
                dim=self.embed_dim,
                depth=self.depths[0],
                num_heads=self.num_heads[0],
                window_size=self.window_size,
                mlp_ratio=self.mlp_ratio,
                qkv_bias=True,
                drop=self.drop_rate,
                attn_drop=self.attn_drop_rate,
                drop_path=dpr[0:self.depths[0]],
                downsample=None
            )
        ])

        # Upsampling to original resolution (240x240 -> 960x960, 4x)
        self.final_upsample = nn.Sequential(
            nn.ConvTranspose2d(self.embed_dim, self.embed_dim // 2, kernel_size=2, stride=2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(self.embed_dim // 2, self.embed_dim // 4, kernel_size=2, stride=2),
            nn.ReLU(inplace=True)
        )

        # Output head
        self.output_head = nn.Conv2d(self.embed_dim // 4, out_channels, kernel_size=1)

        # Normalization layers
        self.norm1 = nn.LayerNorm(self.embed_dim)
        self.norm2 = nn.LayerNorm(self.embed_dim * 2)
        self.norm3 = nn.LayerNorm(self.embed_dim * 4)
        self.norm4 = nn.LayerNorm(self.embed_dim * 8)

    def forward(self, x):
        # Input: (B, 1, 960, 960)
        B = x.shape[0]

        # CNN Backbone
        # Returns [b1, b2, b3, b4] at different scales
        # b1: 240x240x18, b2: 120x120x36, b3: 60x60x72, b4: 30x30x144
        backbone_features = self.backbone(x)

        # Patch embeddings
        e1, H1, W1 = self.patch_embed1(backbone_features[0])  # 240x240x64
        e2, H2, W2 = self.patch_embed2(backbone_features[1])  # 120x120x128
        e3, H3, W3 = self.patch_embed3(backbone_features[2])  # 60x60x256
        e4, H4, W4 = self.patch_embed4(backbone_features[3])  # 30x30x512

        # ENCODER with skip connections
        enc1, H1_down, W1_down = self.encoder1(e1, H1, W1)  # -> 120x120x128
        enc1 = self.norm2(enc1)

        # Concatenate with b2
        enc1_reshaped = enc1.view(B, H1_down, W1_down, -1).permute(0, 3, 1, 2)
        e2_reshaped = e2.view(B, H2, W2, -1).permute(0, 3, 1, 2)
        enc2_input = torch.cat([enc1_reshaped, e2_reshaped], dim=1)
        enc2_input = enc2_input.flatten(2).transpose(1, 2)
        enc2_input = self.fusion2(enc2_input)

        enc2, H2_down, W2_down = self.encoder2(enc2_input, H2, W2)  # -> 60x60x256
        enc2 = self.norm3(enc2)

        # Concatenate with b3
        enc2_reshaped = enc2.view(B, H2_down, W2_down, -1).permute(0, 3, 1, 2)
        e3_reshaped = e3.view(B, H3, W3, -1).permute(0, 3, 1, 2)
        enc3_input = torch.cat([enc2_reshaped, e3_reshaped], dim=1)
        enc3_input = enc3_input.flatten(2).transpose(1, 2)
        enc3_input = self.fusion3(enc3_input)

        enc3, H3_down, W3_down = self.encoder3(enc3_input, H3, W3)  # -> 30x30x512
        enc3 = self.norm4(enc3)

        # Concatenate with b4
        enc3_reshaped = enc3.view(B, H3_down, W3_down, -1).permute(0, 3, 1, 2)
        e4_reshaped = e4.view(B, H4, W4, -1).permute(0, 3, 1, 2)
        enc4_input = torch.cat([enc3_reshaped, e4_reshaped], dim=1)
        enc4_input = enc4_input.flatten(2).transpose(1, 2)
        enc4_input = self.fusion4(enc4_input)

        # BOTTLENECK
        bottleneck, H_bottle, W_bottle = self.bottleneck(enc4_input, H4, W4)  # 30x30x512
        bottleneck = self.norm4(bottleneck)

        # Apply dropout
        bottleneck_2d = bottleneck.view(B, H_bottle, W_bottle, -1).permute(0, 3, 1, 2)
        bottleneck_2d = self.dropout(bottleneck_2d)
        bottleneck = bottleneck_2d.flatten(2).transpose(1, 2)

        # DECODER
        # Decoder 4: 30x30 -> 60x60
        dec4, H_dec4, W_dec4 = self.decoder4[0](bottleneck, H_bottle, W_bottle)
        dec4_cat = torch.cat([dec4, enc2], dim=-1)
        dec4_cat = self.decoder4[1](dec4_cat)
        dec4, _, _ = self.decoder4[2](dec4_cat, H_dec4, W_dec4)

        # Decoder 3: 60x60 -> 120x120
        dec3, H_dec3, W_dec3 = self.decoder3[0](dec4, H_dec4, W_dec4)
        dec3_cat = torch.cat([dec3, enc1], dim=-1)
        dec3_cat = self.decoder3[1](dec3_cat)
        dec3, _, _ = self.decoder3[2](dec3_cat, H_dec3, W_dec3)

        # Decoder 2: 120x120 -> 240x240
        dec2, H_dec2, W_dec2 = self.decoder2[0](dec3, H_dec3, W_dec3)
        dec2_cat = torch.cat([dec2, e1], dim=-1)
        dec2_cat = self.decoder2[1](dec2_cat)
        dec2, _, _ = self.decoder2[2](dec2_cat, H_dec2, W_dec2)

        # Reshape to 2D
        dec2_2d = dec2.view(B, H_dec2, W_dec2, -1).permute(0, 3, 1, 2)  # B, C, H, W

        # Upsample to 960x960
        out = self.final_upsample(dec2_2d)  # B, C/4, 960, 960

        # Final output
        out = self.output_head(out)  # B, 1, 960, 960

        return out
    

class RMSELoss(nn.Module):
    """Root Mean Squared Error Loss (includes background)"""
    def __init__(self, eps=1e-8):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
    
    def forward(self, pred, target):
        return torch.sqrt(self.mse(pred, target) + self.eps)


class MaskedRMSELoss(nn.Module):
    """
    RMSE loss that ignores background pixels (depth == 0)
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        # Create mask for non-zero pixels
        mask = (target > 0).float()
        
        # Compute squared error only on masked pixels
        squared_error = (pred - target) ** 2
        masked_squared_error = squared_error * mask
        
        # Mean over valid pixels
        mse = masked_squared_error.sum() / mask.sum().clamp(min=self.eps)
        
        # RMSE
        return torch.sqrt(mse + self.eps)
