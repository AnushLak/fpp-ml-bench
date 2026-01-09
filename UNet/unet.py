"""
UNet for Single-Shot Fringe Projection Profilometry
Input: 960x960 grayscale fringe pattern
Output: 960x960 depth map
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3x3 convolutions with BatchNorm and ReLU"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """Downsampling block: DoubleConv + MaxPool"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        skip = self.conv(x)  # Save for skip connection
        x = self.pool(skip)
        return skip, x


class UpSample(nn.Module):
    """Upsampling block: ConvTranspose + Concat + DoubleConv"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        # Concatenate with skip connection
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetFPP(nn.Module):
    """
    UNet for Fringe Projection Profilometry
    
    Architecture:
    - Input: (B, 1, 960, 960) - Grayscale fringe pattern
    - Output: (B, 1, 960, 960) - Depth map
    
    Pathway:
    960 -> 480 -> 240 -> 120 -> 60 (bottleneck)
    60 -> 120 -> 240 -> 480 -> 960
    """
    
    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.5):
        super().__init__()
        
        # ENCODER
        self.down1 = DownSample(in_channels, 64)   # 960 -> 480
        self.down2 = DownSample(64, 128)           # 480 -> 240
        self.down3 = DownSample(128, 256)          # 240 -> 120
        self.down4 = DownSample(256, 512)          # 120 -> 60
        
        # BOTTLENECK
        self.bottleneck = DoubleConv(512, 1024)    # 60x60
        self.dropout = nn.Dropout2d(p=dropout_rate)
        
        # DECODER
        self.up1 = UpSample(1024, 512)             # 60 -> 120
        self.up2 = UpSample(512, 256)              # 120 -> 240
        self.up3 = UpSample(256, 128)              # 240 -> 480
        self.up4 = UpSample(128, 64)               # 480 -> 960
        
        # OUTPUT
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights using He initialization"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # ENCODER with skip connections
        skip1, x = self.down1(x)  # skip: 960x960x64,  x: 480x480x64
        skip2, x = self.down2(x)  # skip: 480x480x128, x: 240x240x128
        skip3, x = self.down3(x)  # skip: 240x240x256, x: 120x120x256
        skip4, x = self.down4(x)  # skip: 120x120x512, x: 60x60x512
        
        # BOTTLENECK
        x = self.bottleneck(x)    # 60x60x1024
        x = self.dropout(x)
        
        # DECODER with skip connections
        x = self.up1(x, skip4)    # 120x120x512
        x = self.up2(x, skip3)    # 240x240x256
        x = self.up3(x, skip2)    # 480x480x128
        x = self.up4(x, skip1)    # 960x960x64
        
        # OUTPUT
        x = self.out(x)           # 960x960x1
        
        return x


class MaskedRMSELoss(nn.Module):
    """
    RMSE loss that ignores background pixels (depth=0)
    
    This is critical for FPP where not all pixels have valid depth
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        """
        Args:
            pred: (B, 1, H, W) predicted depth
            target: (B, 1, H, W) ground truth depth (0 = background)
        """
        # Create mask: valid pixels where target > 0
        mask = (target > 0).float()
        
        # Compute squared error only on valid pixels
        squared_error = (pred - target) ** 2
        masked_squared_error = squared_error * mask
        
        # Mean over valid pixels
        num_valid = mask.sum().clamp(min=self.eps)
        mse = masked_squared_error.sum() / num_valid
        
        return torch.sqrt(mse + self.eps)


if __name__ == "__main__":
    # Test the model
    model = UNetFPP(in_channels=1, out_channels=1, dropout_rate=0.5)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Test forward pass
    x = torch.randn(2, 1, 960, 960)
    y = model(x)
    print(f"\nInput shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    
    # Test loss
    criterion = MaskedRMSELoss()
    target = torch.randn(2, 1, 960, 960).abs()  # Positive depth values
    loss = criterion(y, target)
    print(f"\nLoss: {loss.item():.6f}")
