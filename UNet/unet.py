"""
UNet for Single-Shot Fringe Projection Profilometry
Input: 960x960 grayscale fringe pattern
Output: 960x960 depth map
"""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """Two 3×3 convolutions with InstanceNorm + ReLU."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm2d(out_channels, affine=True),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class DownSample(nn.Module):
    """Downsampling block: DoubleConv + MaxPool."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        skip = self.conv(x)
        x = self.pool(skip)
        return skip, x


class UpSample(nn.Module):
    """Upsampling block: ConvTranspose + Concat + DoubleConv."""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([skip, x], dim=1)
        return self.conv(x)


class UNetFPP(nn.Module):
    """
    UNet for Fringe Projection Profilometry
    Input:  (B, 1, 960, 960)
    Output: (B, 1, 960, 960)
    """

    def __init__(self, in_channels=1, out_channels=1, dropout_rate=0.5):
        super().__init__()

        # ENCODER (original widths)
        self.down1 = DownSample(in_channels, 64)
        self.down2 = DownSample(64, 128)
        self.down3 = DownSample(128, 256)
        self.down4 = DownSample(256, 512)

        # BOTTLENECK
        self.bottleneck = DoubleConv(512, 1024)
        self.dropout = nn.Dropout2d(p=dropout_rate)

        # DECODER
        self.up1 = UpSample(1024, 512)
        self.up2 = UpSample(512, 256)
        self.up3 = UpSample(256, 128)
        self.up4 = UpSample(128, 64)

        # OUTPUT
        self.out = nn.Conv2d(64, out_channels, kernel_size=1)

        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        skip1, x = self.down1(x)
        skip2, x = self.down2(x)
        skip3, x = self.down3(x)
        skip4, x = self.down4(x)

        x = self.bottleneck(x)
        x = self.dropout(x)

        x = self.up1(x, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        return self.out(x)


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


class HybridRMSELoss(nn.Module):
    """
    Masked RMSE + weak global RMSE anchor.
    Prevents scale drift while still ignoring background pixels.
    """
    def __init__(self, alpha=0.9, eps=1e-8):
        super().__init__()
        self.alpha = alpha  # weight for masked RMSE
        self.eps = eps

    def forward(self, pred, target):
        # Masked RMSE (valid pixels only)
        mask = (target > 0).float()

        diff = (pred - target) * mask
        mse_masked = (diff * diff).sum() / mask.sum().clamp(min=self.eps)
        masked_rmse = torch.sqrt(mse_masked + self.eps)

        # Global RMSE (weak anchor)
        mse_global = ((pred - target) ** 2).mean()
        global_rmse = torch.sqrt(mse_global + self.eps)

        # Combined loss
        return self.alpha * masked_rmse + (1 - self.alpha) * global_rmse
    

class HybridMaskedRMSEWithMaskedL1(nn.Module):
    """
    Masked RMSE + weak global RMSE + masked L1.
    This sharpens interior geometry without letting background dominate.
    """
    def __init__(self, alpha=0.8, lambda_l1=0.05, eps=1e-8):
        super().__init__()
        self.alpha = alpha
        self.lambda_l1 = lambda_l1
        self.eps = eps

    def forward(self, pred, target):
        # Mask for valid object pixels
        mask = (target > 0).float()

        # Masked RMSE (object only)
        diff_masked = (pred - target) * mask
        mse_masked = (diff_masked * diff_masked).sum() / mask.sum().clamp(min=self.eps)
        masked_rmse = torch.sqrt(mse_masked + self.eps)

        # Global RMSE (all pixels)
        mse_global = ((pred - target) ** 2).mean()
        global_rmse = torch.sqrt(mse_global + self.eps)

        # Masked L1 (object only)
        masked_l1 = (diff_masked.abs().sum() / mask.sum().clamp(min=self.eps))

        # Combined loss
        loss = (
            self.alpha * masked_rmse +
            (1 - self.alpha) * global_rmse +
            self.lambda_l1 * masked_l1
        )

        return loss

class L1Loss(nn.Module):
    """
    Simple L1 loss (Mean Absolute Error).
    Computes the mean absolute difference between prediction and target.
    """
    def __init__(self):
        super().__init__()

    def forward(self, pred, target):
        return torch.abs(pred - target).mean()

class MaskedL1Loss(nn.Module):
    """
    Masked L1 loss that only considers valid pixels (where target > 0).
    Ignores background pixels in the loss calculation.
    """
    def __init__(self, eps=1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred, target):
        # Create mask: valid pixels where target > 0
        mask = (target > 0).float()

        # Compute L1 only on valid pixels
        diff = torch.abs(pred - target) * mask

        # Mean over valid pixels
        num_valid = mask.sum().clamp(min=self.eps)
        masked_l1 = diff.sum() / num_valid

        return masked_l1
    
class HybridL1Loss(nn.Module):
    """
    Masked L1 + weak global L1 anchor.
    Prevents scale drift while still ignoring background pixels.
    Uses L1 (absolute difference) instead of RMSE.
    """
    def __init__(self, alpha=0.9, eps=1e-8):
        super().__init__()
        self.alpha = alpha  # weight for masked L1
        self.eps = eps

    def forward(self, pred, target):
        # Masked L1 (valid pixels only)
        mask = (target > 0).float()

        diff = torch.abs(pred - target) * mask
        masked_l1 = diff.sum() / mask.sum().clamp(min=self.eps)

        # Global L1 (weak anchor)
        global_l1 = torch.abs(pred - target).mean()

        # Combined loss
        return self.alpha * masked_l1 + (1 - self.alpha) * global_l1


# if __name__ == "__main__":
#     # Test the model
#     model = UNetFPP(in_channels=1, out_channels=1, dropout_rate=0.0)
    
#     # Count parameters
#     total_params = sum(p.numel() for p in model.parameters())
#     trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
#     print(f"Total parameters: {total_params:,}")
#     print(f"Trainable parameters: {trainable_params:,}")
    
#     # Test forward pass
#     x = torch.randn(2, 1, 960, 960)
#     y = model(x)
#     print(f"\nInput shape: {x.shape}")
#     print(f"Output shape: {y.shape}")
    
#     # Test loss
#     criterion = MaskedRMSELoss()
#     target = torch.randn(2, 1, 960, 960).abs()  # Positive depth values
#     loss = criterion(y, target)
#     print(f"\nLoss: {loss.item():.6f}")