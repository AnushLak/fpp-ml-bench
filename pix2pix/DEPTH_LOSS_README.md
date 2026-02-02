# Depth Loss Functions for pix2pixHD

This document describes the depth loss function implementations added to pix2pixHD for fringe-to-depth translation.

## Overview

Three depth loss functions are now available:
1. **L1 Loss** (MAE - Mean Absolute Error)
2. **RMSE Loss** (Root Mean Squared Error)
3. **Masked RMSE Loss** (RMSE computed only on object pixels)

## Loss Function Implementations

### 1. L1 Loss (default)
```python
loss = torch.mean(|pred - target|)
```
- Standard Mean Absolute Error
- Treats all pixels equally
- Good for general depth estimation

### 2. RMSE Loss
```python
loss = sqrt(torch.mean((pred - target)^2))
```
- Root Mean Squared Error
- Penalizes larger errors more heavily
- Better for reducing outliers

### 3. Masked RMSE Loss (Recommended)
```python
mask = (target > 0)  # Only object pixels
loss = sqrt(sum(mask * (pred - target)^2) / sum(mask))
```
- RMSE computed only on object pixels (where ground truth > 0)
- Ignores background pixels
- **Most suitable for FPP** as it focuses learning on the object of interest
- Based on UNet/ResUNet masked RMSE implementation

## Usage

### Command Line

Specify the loss type when training:

```bash
# Using L1 loss
python train.py --depth_loss_type l1 ...

# Using RMSE loss
python train.py --depth_loss_type rmse ...

# Using Masked RMSE loss (Recommended)
python train.py --depth_loss_type masked_rmse ...
```

### Bash Script

The `pix2pix.sh` script has been updated with the `--depth_loss_type` parameter:

```bash
python -u train.py \
    --name fringe2depth_exp \
    --depth_loss_type masked_rmse \
    --dataset_mode fringe_depth \
    ...
```

To use a different loss, simply change the `--depth_loss_type` value:
- `l1` → Standard L1 loss
- `rmse` → RMSE loss
- `masked_rmse` → Masked RMSE loss (focuses on object only)

## Checkpoint Naming

The loss type is **automatically appended** to the checkpoint folder name for easy differentiation:

- `--name fringe2depth_exp --depth_loss_type l1` → `checkpoints/fringe2depth_exp_l1/`
- `--name fringe2depth_exp --depth_loss_type rmse` → `checkpoints/fringe2depth_exp_rmse/`
- `--name fringe2depth_exp --depth_loss_type masked_rmse` → `checkpoints/fringe2depth_exp_masked_rmse/`

This allows you to train with different loss functions simultaneously without overwriting checkpoints.

## Loss Logging

The loss will be displayed with the appropriate name in training logs:

- L1 loss: `G_L1: 0.123`
- RMSE loss: `G_RMSE: 0.145`
- Masked RMSE loss: `G_MASKED_RMSE: 0.167`

## Example Training Commands

### Train with L1 Loss
```bash
sbatch pix2pix.sh  # Edit script to set --depth_loss_type l1
```

### Train with RMSE Loss
```bash
python train.py \
    --name fringe2depth_exp \
    --dataset_mode fringe_depth \
    --depth_loss_type rmse \
    --lambda_L1 10 \
    ...
```

### Train with Masked RMSE Loss (Recommended)
```bash
python train.py \
    --name fringe2depth_exp \
    --dataset_mode fringe_depth \
    --depth_loss_type masked_rmse \
    --lambda_L1 10 \
    ...
```

## Recommendation

For Fringe Projection Profilometry (FPP), **`masked_rmse`** is recommended because:
1. It focuses learning on the object of interest (ignores background)
2. Matches the evaluation metric used in UNet/ResUNet
3. Prevents the model from "cheating" by predicting zeros for background
4. More robust to imbalanced object/background ratios

## Files Modified

1. **`options/train_options.py`** - Added `--depth_loss_type` argument
2. **`models/pix2pixHD_model.py`** - Added RMSELoss and MaskedRMSELoss classes
3. **`options/base_options.py`** - Auto-append loss type to experiment name
4. **`pix2pix.sh`** - Updated with `--depth_loss_type` parameter and documentation

## Technical Details

### Loss Weight
The `--lambda_L1` parameter controls the weight of the depth loss (regardless of type):
```bash
--lambda_L1 10  # Multiplies the loss by 10
```

### Masked RMSE Implementation
- Threshold: 0.0 (pixels with depth > 0 are considered object)
- Input range: [-1, 1] (converted from [0, 1] in dataset)
- Epsilon: 1e-8 (prevents division by zero)

### Integration with Other Losses
The depth loss is combined with other losses:
```
Total_Loss = G_GAN + G_GAN_Feat + Depth_Loss + G_Grad + G_SI
```

Where `Depth_Loss` can be L1, RMSE, or Masked RMSE based on `--depth_loss_type`.
