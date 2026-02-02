import torch
import torch.nn as nn

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