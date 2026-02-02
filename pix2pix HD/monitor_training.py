import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os

def plot_losses(json_path, save_dir):
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    if not data:
        print("No data to plot yet")
        return
    
    # Extract data
    steps = [d['step'] for d in data]
    epochs = [d.get('epoch', 0) for d in data]
    
    # Get unique epochs for x-axis
    unique_epochs = sorted(list(set(epochs)))
    
    # Generator and discriminator totals
    g_total = [d.get('G_total', 0) for d in data]
    d_total = [d.get('D_total', 0) for d in data]
    
    # Depth-specific losses
    g_l1 = [d.get('G_L1', 0) for d in data]
    g_grad = [d.get('G_Grad', 0) for d in data]
    g_si = [d.get('G_SI', 0) for d in data]
    g_gan = [d.get('G_GAN', 0) for d in data]
    
    # Create plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Total losses
    axes[0, 0].plot(steps, g_total, label='Generator Total', alpha=0.7)
    axes[0, 0].plot(steps, d_total, label='Discriminator Total', alpha=0.7)
    axes[0, 0].set_xlabel('Steps')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Total Losses')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # GAN losses
    axes[0, 1].plot(steps, g_gan, label='GAN Loss', color='orange', alpha=0.7)
    axes[0, 1].set_xlabel('Steps')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].set_title('GAN Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # L1 Loss
    axes[0, 2].plot(steps, g_l1, label='L1 Loss', color='green', alpha=0.7)
    axes[0, 2].set_xlabel('Steps')
    axes[0, 2].set_ylabel('Loss')
    axes[0, 2].set_title('L1 Reconstruction Loss')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Gradient Loss
    axes[1, 0].plot(steps, g_grad, label='Gradient Loss', color='red', alpha=0.7)
    axes[1, 0].set_xlabel('Steps')
    axes[1, 0].set_ylabel('Loss')
    axes[1, 0].set_title('Gradient Loss (Edge Preservation)')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Scale-Invariant Loss
    axes[1, 1].plot(steps, g_si, label='Scale-Invariant Loss', color='purple', alpha=0.7)
    axes[1, 1].set_xlabel('Steps')
    axes[1, 1].set_ylabel('Loss')
    axes[1, 1].set_title('Scale-Invariant Loss')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Loss over epochs (averaged)
    if unique_epochs and unique_epochs[-1] > 0:
        epoch_losses = {}
        for epoch in unique_epochs:
            if epoch > 0:
                epoch_data = [d for d in data if d.get('epoch') == epoch]
                if epoch_data:
                    epoch_losses[epoch] = {
                        'G_total': np.mean([d.get('G_total', 0) for d in epoch_data]),
                        'D_total': np.mean([d.get('D_total', 0) for d in epoch_data]),
                    }
        
        if epoch_losses:
            epochs_list = sorted(epoch_losses.keys())
            g_epoch = [epoch_losses[e]['G_total'] for e in epochs_list]
            d_epoch = [epoch_losses[e]['D_total'] for e in epochs_list]
            
            axes[1, 2].plot(epochs_list, g_epoch, 'o-', label='Gen Avg', alpha=0.7)
            axes[1, 2].plot(epochs_list, d_epoch, 's-', label='Disc Avg', alpha=0.7)
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('Average Loss')
            axes[1, 2].set_title('Average Loss per Epoch')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'loss_curves.png'), dpi=100)
    plt.show()
    
    # Print latest losses
    if data:
        latest = data[-1]
        print(f"\nLatest losses (Step {latest['step']}, Epoch {latest.get('epoch', 'N/A')}):")
        print(f"  Generator Total: {latest.get('G_total', 0):.6f}")
        print(f"  Discriminator Total: {latest.get('D_total', 0):.6f}")
        print(f"  G_GAN: {latest.get('G_GAN', 0):.6f}")
        print(f"  G_L1: {latest.get('G_L1', 0):.6f}")
        print(f"  G_Grad: {latest.get('G_Grad', 0):.6f}")
        print(f"  G_SI: {latest.get('G_SI', 0):.6f}")
        print(f"  D_real: {latest.get('D_real', 0):.6f}")
        print(f"  D_fake: {latest.get('D_fake', 0):.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', type=str, default='fringe2depth_exp', help='experiment name')
    parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='checkpoints directory')
    args = parser.parse_args()
    
    json_path = os.path.join(args.checkpoints_dir, args.name, 'logs', 'losses.json')
    save_dir = os.path.join(args.checkpoints_dir, args.name, 'logs')
    
    if os.path.exists(json_path):
        plot_losses(json_path, save_dir)
    else:
        print(f"Loss file not found: {json_path}")
        print(f"Make sure training has started and losses are being logged.")