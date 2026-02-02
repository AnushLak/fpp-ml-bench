import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataLoader(opt):
    data_loader = CustomDatasetDataLoader()
    data_loader.initialize(opt)
    return data_loader

class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)

        # Create dataset based on dataset_type (for FPP depth estimation)
        # Check if dataset_type is one of our normalization types
        if hasattr(opt, 'dataset_type') and opt.dataset_type in ['_raw', '_global_normalized', '_individual_normalized']:
            from data.fringe_depth_dataset import FringeDepthDataset
            self.dataset = FringeDepthDataset()
        else:
            # Fallback to original dataset_mode if present
            dataset_mode = getattr(opt, 'dataset_mode', 'aligned')
            if dataset_mode == 'aligned':
                from data.aligned_dataset import AlignedDataset
                self.dataset = AlignedDataset()
            elif dataset_mode == 'unaligned':
                from data.unaligned_dataset import UnalignedDataset
                self.dataset = UnalignedDataset()
            else:
                raise ValueError(f"Dataset type [{getattr(opt, 'dataset_type', 'not set')}] or mode [{dataset_mode}] not recognized.")

        print("dataset [%s] was created" % (self.dataset.name()))
        self.dataset.initialize(opt)

        # Create dataloader
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=not opt.serial_batches,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return min(len(self.dataset), self.opt.max_dataset_size)