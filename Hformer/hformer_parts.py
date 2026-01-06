import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
import math

# ============================================================================
# ATTENTION MECHANISMS
# ============================================================================

class Attention(nn.Module):
    """
    Attention module with relative position encoding
    Supports both IPSA (Inner-Patch Self-Attention) and CPSA (Cross-Patch Self-Attention)
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.,
                 rpe=True, ws=1):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.rpe = rpe
        self.ws = ws

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # Relative position encoding
        if self.rpe:
            self.relative_position_bias_table = nn.Parameter(
                torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
            )
            coords_h = torch.arange(ws)
            coords_w = torch.arange(ws)
            coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
            coords_flatten = torch.flatten(coords, 1)
            relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
            relative_coords = relative_coords.permute(1, 2, 0).contiguous()
            relative_coords[:, :, 0] += ws - 1
            relative_coords[:, :, 1] += ws - 1
            relative_coords[:, :, 0] *= 2 * ws - 1
            relative_position_index = relative_coords.sum(-1)
            self.register_buffer("relative_position_index", relative_position_index)
            nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        if self.rpe:
            relative_position_bias = self.relative_position_bias_table[
                self.relative_position_index.view(-1)
            ].view(self.ws * self.ws, self.ws * self.ws, -1)
            relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
            attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-forward network"""
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DropPath(nn.Module):
    """Stochastic depth (drop path) regularization"""
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


class CATBlock(nn.Module):
    """
    Cross Attention Transformer Block
    Combines attention mechanism with MLP
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.,
                 attn_drop=0., drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,
                 rpe=True, ws=7):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop,
            proj_drop=drop, rpe=rpe, ws=ws
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim,
                      act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


# ============================================================================
# PATCH OPERATIONS
# ============================================================================

def window_partition(x, window_size):
    """
    Partition feature map into windows
    Args:
        x: (B, H, W, C)
        window_size: int
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Reverse window partition
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size: int
        H: Height of image
        W: Width of image
    Returns:
        x: (B, H, W, C)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


class PatchEmbedding(nn.Module):
    """Convert feature map to patch embeddings"""
    def __init__(self, in_channels, embed_dim, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size,
                             stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # B, C, H, W
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, H*W, C
        x = self.norm(x)
        return x, H, W


class PatchProjection(nn.Module):
    """Downsample by merging patches"""
    def __init__(self, dim, out_dim):
        super().__init__()
        self.norm = nn.LayerNorm(4 * dim)
        self.reduction = nn.Linear(4 * dim, out_dim, bias=False)

    def forward(self, x, H, W):
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        x = x.view(B, H, W, C)

        # Pad if needed
        pad_input = (H % 2 == 1) or (W % 2 == 1)
        if pad_input:
            x = F.pad(x, (0, 0, 0, W % 2, 0, H % 2))

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C
        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C

        x = x.view(B, -1, 4 * C)  # B H/2*W/2 4*C
        x = self.norm(x)
        x = self.reduction(x)

        return x, H // 2, W // 2


class PatchExpand(nn.Module):
    """Upsample by expanding patches"""
    def __init__(self, dim, out_dim, scale=2):
        super().__init__()
        self.dim = dim
        self.expand = nn.Linear(dim, scale * scale * out_dim, bias=False)
        self.norm = nn.LayerNorm(out_dim)
        self.scale = scale

    def forward(self, x, H, W):
        B, L, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b (h w) (p1 p2 c) -> b (h p1) (w p2) c',
                     h=H, w=W, p1=self.scale, p2=self.scale)
        x = x.view(B, -1, x.shape[-1])
        x = self.norm(x)

        return x, H * self.scale, W * self.scale


# ============================================================================
# CAT LAYER
# ============================================================================

class CATLayer(nn.Module):
    """
    Cross Attention Transformer Layer
    Contains multiple CATBlocks with IPSA and CPSA
    """
    def __init__(self, dim, depth, num_heads, window_size=7, mlp_ratio=4.,
                 qkv_bias=True, drop=0., attn_drop=0., drop_path=0.,
                 downsample=None):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.window_size = window_size

        # Build blocks
        self.blocks = nn.ModuleList([
            CATBlock(
                dim=dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                qkv_bias=qkv_bias,
                drop=drop,
                attn_drop=attn_drop,
                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
                rpe=True,
                ws=window_size
            )
            for i in range(depth)
        ])

        # Downsample layer
        self.downsample = downsample

    def forward(self, x, H, W):
        for blk in self.blocks:
            x = blk(x)

        if self.downsample is not None:
            x, H, W = self.downsample(x, H, W)

        return x, H, W
