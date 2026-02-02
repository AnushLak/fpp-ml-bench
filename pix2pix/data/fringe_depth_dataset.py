import os.path
import sys
from pathlib import Path
from data.base_dataset import BaseDataset, get_params, get_transform
from PIL import Image
import numpy as np
import torch
import scipy.io as sio

# Add parent directory to path to import root dataset.py
sys.path.append(str(Path(__file__).parent.parent.parent))
from dataset import FringeFPPDataset

class FringeDepthDataset(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        # Setup paths based on dataset_type
        phase = opt.phase
        dataset_type = getattr(opt, 'dataset_type', '_individual_normalized')

        data_root = Path(f'/work/arpawar/anushlak/SPIE-PW/training_data_depth{dataset_type}')
        fringe_dir = data_root / phase / 'fringe'
        depth_dir = data_root / phase / 'depth'

        # Use the standardized FringeFPPDataset
        self.fpp_dataset = FringeFPPDataset(fringe_dir, depth_dir, dataset_type)
        self.dataset_size = len(self.fpp_dataset)
        
    def __getitem__(self, index):
        # Get data from standardized FringeFPPDataset
        fringe, depth, filename = self.fpp_dataset[index]

        # fringe and depth are already tensors in shape (1, H, W) and normalized [0, 1]
        A_array = fringe.squeeze(0).numpy()  # (H, W)
        B_array = depth.squeeze(0).numpy()   # (H, W)

        # Resize to target size if needed
        if A_array.shape != (self.opt.fineSize, self.opt.fineSize):
            A_array = self.resize_array(A_array, self.opt.fineSize, is_depth=False)
            B_array = self.resize_array(B_array, self.opt.fineSize, is_depth=True)

        # Convert to tensors (add channel dimension)
        A_tensor = torch.from_numpy(A_array).unsqueeze(0)  # Shape: [1, H, W]
        B_tensor = torch.from_numpy(B_array).unsqueeze(0)  # Shape: [1, H, W]

        # Convert from [0, 1] to [-1, 1] for the generator (pix2pixHD expects this range)
        A_tensor = A_tensor * 2.0 - 1.0
        B_tensor = B_tensor * 2.0 - 1.0

        # Create dummy instance map with same spatial dimensions (required by pix2pixHD)
        # Will be ignored if --no_instance flag is set
        inst_tensor = torch.zeros(1, A_array.shape[0], A_array.shape[1])

        return {'label': A_tensor, 'inst': inst_tensor, 'image': B_tensor,
                'feat': torch.zeros(1), 'path': filename}
    
    def resize_array(self, array, size, is_depth=False):
        from PIL import Image
        if is_depth:
            # For depth from .mat files: already normalized, preserve precision
            # Scale to uint16 range for resizing to maintain precision
            max_val = array.max() if array.max() > 0 else 1.0
            img = Image.fromarray((array / max_val * 65535).astype(np.uint16))
            img = img.resize((size, size), Image.BILINEAR)
            return np.array(img).astype(np.float32) / 65535.0 * max_val
        else:
            # For fringe images: uint8 is sufficient
            img = Image.fromarray((array * 255).astype(np.uint8))
            img = img.resize((size, size), Image.BILINEAR)
            return np.array(img).astype(np.float32) / 255.0
    
    def __len__(self):
        return len(self.fpp_dataset)
    
    def name(self):
        return 'FringeDepthDataset'