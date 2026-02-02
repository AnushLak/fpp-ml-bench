# Comprehensive Machine Learning Benchmarking for Fringe Projection Profilometry with Photorealistic Synthetic Data
**Repository: FPP-ML-Bench**
_Accepted to SPIE Photonics West 2026 Conference on Photonic Instrumentation Engineering XIII_

## Overview

This repository provides a standardized benchmarking framework for evaluating deep learning models on single-shot fringe projection profilometry (FPP) depth estimation. FPP is a 3D imaging technique that reconstructs depth maps from projected fringe patterns, enabling high-precision 3D reconstruction for industrial inspection, quality control, and computer vision applications.

The framework implements three state-of-the-art architectures with unified training pipelines, loss functions, and dataset handling:

- **UNet**: Classic encoder-decoder architecture with skip connections
- **Hformer**: Hybrid CNN-Transformer model combining HRNet backbone with transformer encoder-decoder
- **ResUNet**: Residual U-Net architecture with residual blocks for improved gradient flow
- **pix2pixHD**: Conditional GAN with U-Net generator and PatchGAN discriminator

## Features

- 🎯 **Unified Training Pipeline**: Standardized training scripts across all models
- 📊 **6 Loss Functions**: RMSE, Masked RMSE, Hybrid RMSE, L1, Masked L1, Hybrid L1
- 🔄 **3 Normalization Schemes**: Raw, global normalization, individual normalization
- 💾 **Flexible Data Format**: PNG fringe input + MAT depth output

## Repository Structure

```
 FPP-ML-Benchmarking/
  ├── dataset.py              # Common dataloader for all models
  ├── losses.py               # All 6 loss functions
  ├── UNet/
  │   ├── unet.py            # Model architecture
  │   └── train.py           # Training script
  ├── Hformer/
  │   ├── hformer.py         # Model architecture
  │   ├── hformer_parts.py   # Model components
  │   ├── hrnet_backbone.py  # HRNet backbone
  │   └── train.py           # Training script
  ├── ResUNet/
  │   ├── resunet.py         # Model architecture
  │   ├── resunet_parts.py   # Model components
  │   └── train.py           # Training script
  └── pix2pixHD/
      ├── train.py           # Training script
      ├── data/
      │   ├── fringe_depth_dataset.py  # Dataset loader (uses root dataset.py)
      │   └── data_loader.py           # Data loading utilities
      ├── models/
      │   ├── pix2pixHD_model.py       # Generator/Discriminator model
      │   └── networks.py              # Network architectures
      ├── options/
      │   ├── base_options.py          # Base options
      │   └── train_options.py         # Training options (--loss, --alpha, --dataset_type)
      └── util/
          ├── visualizer.py            # Training visualization
          └── util.py                  # Utility functions
```

## Installation

### Requirements

- Python 3.8+
- PyTorch 1.10+
- CUDA 11.0+ (for GPU training)
- SciPy 1.13+ (for depth matrix loading)
- Pillow 11.2+
- tqdm 4.6+ (for training tracking)
- Matplotlib 3.8+

### Setup

```bash
# Clone the repository
git clone https://github.com/AnushLak/FPP-ML-Benchmarking.git
cd FPP-ML-Benchmarking

# Create virtual environment
python -m venv venv # or conda create --name <env_name>
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install numpy scipy pillow tqdm matplotlib
```

## Dataset Structure

The repository expects the following dataset structure:

```
training_data_depth{dataset_type}/
├── train/
│   ├── fringe/          # PNG fringe images
│   │   ├── sample_001.png
│   │   ├── sample_002.png
│   │   └── ...
│   └── depth/           # MAT depth maps
│       ├── sample_001.mat
│       ├── sample_002.mat
│       └── ...
├── val/
│   ├── fringe/
│   └── depth/
└── test/
    ├── fringe/
    └── depth/
```

### Dataset Types

- **`_raw`**: Raw depth values from .mat files (no normalization)
- **`_global_normalized`**: Normalized by global constant (65535)
- **`_individual_normalized`**: Each depth map normalized by its maximum value

Update the data paths in each model's training script if your data is located elsewhere.

## Usage

### Training

All models share the same command-line interface:

```bash
cd <model_name>  # UNet, Hformer, ResUNet or Pix2PixHD
python train.py [OPTIONS]
```

#### Basic Training Examples

**UNet with default settings:**
```bash
cd UNet
python train.py --dataset_type _individual_normalized --loss hybrid_l1 --alpha 0.9
```

**Hformer with masked RMSE loss:**
```bash
cd Hformer
python train.py --dataset_type _global_normalized --loss masked_rmse --batch_size 1 --epochs 1000
```

**ResUNet with custom hyperparameters:**
```bash
cd ResUNet
python train.py --dataset_type _raw --loss hybrid_rmse --alpha 0.7 --lr 5e-5 --dropout 0.3
```

**Pix2PixHD with custom hyperparameters:**
```bash
cd pix2pixHD
python train.py --dataset_type _individual_normalized --loss hybrid_l1 --alpha 0.7 --lr 1e-5 --dropout 0.0
```

#### Training Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--dataset_type` | str | `_individual_normalized` | Dataset normalization: `_raw`, `_global_normalized`, `_individual_normalized` |
| `--loss` | str | `hybrid_l1` | Loss function: `rmse`, `masked_rmse`, `hybrid_rmse`, `l1`, `masked_l1`, `hybrid_l1` |
| `--alpha` | float | 0.9 | Alpha parameter for hybrid losses (0-1), weight for masked component |
| `--batch_size` | int | 2 (UNet), 1 (Hformer), 4 (ResUNet) | Training batch size |
| `--epochs` | int | 1000 (UNet/Hformer), 600 (ResUNet) | Number of training epochs |
| `--lr` | float | 1e-4 | Initial learning rate |
| `--dropout` | float | 0.5 (UNet), 0.0 (Hformer/ResUNet) | Dropout rate |
| `--resume` | str | None | Path to checkpoint to resume training |
| `--num_workers` | int | 4 | Number of data loading workers |
| `--save_every` | int | 10 | Save checkpoint every N epochs |

#### Resume Training

```bash
python train.py --resume checkpoints/best_model.pth
```

### Monitoring Training

Training logs are saved to `logs/training_log.csv` with the following columns:
- `epoch`: Epoch number
- `train_loss`: Training loss
- `val_loss`: Validation loss
- `time_seconds`: Epoch duration
- `learning_rate`: Current learning rate
- `iteration`: Total iteration count

Checkpoints are saved to `checkpoints/`:
- `best_model.pth`: Best validation loss
- `checkpoint_epoch_XXXX.pth`: Periodic checkpoints

## Loss Functions

### Available Loss Functions

1. **RMSE Loss** (`rmse`)
   - Root Mean Squared Error on all pixels
   - Good for general regression

$$
\mathcal{L}_{\text{RMSE}} = \sqrt{\frac{1}{HW}\sum_{u=1}^{W}\sum_{v=1}^{H} (\hat{D}(u,v) - D(u,v))^2 + \epsilon}
$$

     where $\epsilon = 10^{-8}$ ensures numerical stability.

2. **Masked RMSE Loss** (`masked_rmse`)
   - RMSE computed only on valid pixels (depth > 0)
   - Ignores background, focuses on objects

3. **Hybrid RMSE Loss** (`hybrid_rmse`)
   - Combines masked RMSE with weak global RMSE anchor
   - Formula: `α × masked_rmse + (1-α) × global_rmse`
   - Prevents scale drift while prioritizing objects

4. **L1 Loss** (`l1`)
   - Mean Absolute Error on all pixels
   - More robust to outliers than RMSE

5. **Masked L1 Loss** (`masked_l1`)
   - L1 computed only on valid pixels
   - Ignores background

6. **Hybrid L1 Loss** (`hybrid_l1`) - **Recommended**
   - Combines masked L1 with weak global L1 anchor
   - Formula: `α × masked_l1 + (1-α) × global_l1`
   - Best balance between accuracy and stability

### Choosing Alpha

The `alpha` parameter controls the weight between masked and global components in hybrid losses:
- **α = 1.0**: Pure masked loss (ignores background completely)
- **α = 0.9**: Strong focus on objects, weak background constraint (default)
- **α = 0.7**: Balanced between objects and background
- **α = 0.5**: Equal weight
- **α = 0.0**: Pure global loss (treats all pixels equally)

**Recommendation**: Start with `α = 0.7` for hybrid losses.

## Model Architectures

### UNet
- **Parameters**: ~31M
- **Input**: 960×960 grayscale fringe image
- **Output**: 960×960 depth map
- **Architecture**: Classic U-Net with InstanceNorm and skip connections
- **Best for**: Baseline performance, fast training

### Hformer
- **Parameters**: ~5M
- **Input**: 960×960 grayscale fringe image
- **Output**: 960×960 depth map
- **Architecture**: HRNet backbone + Transformer encoder-decoder
- **Best for**: High-accuracy applications, complex fringe patterns

### ResUNet
- **Parameters**: ~89M
- **Input**: 960×960 grayscale fringe image
- **Output**: 960×960 depth map
- **Architecture**: U-Net with residual blocks and batch normalization
- **Best for**: Balance between accuracy and efficiency

## Training Tips

1. **Start with default settings**: They work well for most cases
2. **Use `hybrid_l1` loss** with `alpha=0.7` as baseline
3. **Monitor validation loss**: Training should converge within 100-200 epochs
4. **Adjust learning rate**: If loss plateaus early, try reducing initial LR
5. **Try different normalizations**: `_individual_normalized` usually works best
6. **Use larger batch size** if GPU memory allows (faster training)

## Optimization

All models use:
- **Optimizer**: RMSprop with weight decay 1e-5 (you can also use ADAM)
- **Dropout Rate**: Dropout rate of 0.0 for initial learning
- **Learning Rate Scheduler**: ReduceLROnPlateau (factor=0.1, patience=10)
- **Random Seed**: 42 (for reproducibility)

## Troubleshooting

**Out of Memory (OOM) errors:**
- Reduce batch size: `--batch_size 1`
- Reduce number of workers: `--num_workers 2`
- Use gradient checkpointing (modify model code)

**Training loss not decreasing:**
- Check data normalization matches dataset type
- Verify .mat files contain valid depth data
- Try different loss function (e.g., `masked_l1`)
- Reduce learning rate: `--lr 1e-6`

**Import errors:**
- Ensure you're running from model directory (cd UNet/Hformer/ResUNet)
- Check Python path includes parent directory
- Verify all dependencies installed

## Citation

If you use this code in your research, please cite the following papers:

```bibtex
1) @article{lakshman2026comprehensive,
  title={Comprehensive Machine Learning Benchmarking for Fringe Projection Profilometry with Photorealistic Synthetic Data},
  author={Lakshman S, Anush and Haroon, Adam and Li, Beiwen},
  journal={arXiv preprint arXiv:2601.08900},
  year={2026}
}

2) @article{haroon2025virtus,
  title={VIRTUS-FPP: virtual sensor modeling for fringe projection profilometry in NVIDIA Isaac Sim},
  author={Haroon, Adam and Lakshman, Anush and Balasubramaniam, Badrinath and Li, Beiwen},
  journal={arXiv preprint arXiv:2509.22685},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- UNet implementation based on [Ronneberger et al., 2015]
- Hformer architecture inspired by [Zhu et al., 2022] and transformer architectures
- ResUNet design follows residual learning principles from [He et al., 2016]

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## Contact

For questions or issues, please open an issue on GitHub or contact [anushlak@iastate.edu] OR [aharoon@iastate.edu].

---

**Note**: This is a research repository. For production use, additional validation and testing is recommended.
