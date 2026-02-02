import os
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import numpy as np
from PIL import Image
from util import util

opt = TestOptions().parse(save=False)
opt.nThreads = 1
opt.batchSize = 1
opt.serial_batches = True
opt.no_flip = True
opt.dataset_mode = 'fringe_depth'
opt.no_instance = True  # No instance maps for fringe-to-depth translation

# Set which dataset split to use: 'train', 'val', or 'test'
opt.phase = 'test'  # Change to 'val' or 'train' if needed

# Set image sizes to 960x960 (must match training configuration)
opt.loadSize = 960
opt.fineSize = 960
opt.input_nc = 1
opt.output_nc = 1
opt.label_nc = 0

# Load best model instead of latest (can also pass --which_epoch via command line)
# opt.which_epoch = 'best'  # or use a specific epoch like '100'

out_dir = os.path.join(opt.results_dir, opt.name)
os.makedirs(out_dir, exist_ok=True)

print(f"Running inference on '{opt.phase}' dataset")
print(f"Model: {opt.name}, Epoch: {opt.which_epoch}")
print(f"Output directory: {out_dir}")

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
print(f"Loaded {len(dataset)} images from dataset")

model = create_model(opt)
model.eval()

print(f"\nGenerating depth predictions...")
for i, data in enumerate(dataset):
    if i >= opt.how_many:
        break

    generated = model.inference(data['label'], data['inst'], data['image'])

    # Convert back to depth values
    depth_generated = (generated[0].detach().cpu().numpy() + 1) / 2.0  # [-1,1] to [0,1]
    depth_generated = (depth_generated * 65535).astype(np.uint16)

    # Save as uint16 PNG
    img_path = data['path'][0]
    short_path = os.path.basename(img_path)
    name = os.path.splitext(short_path)[0]

    if depth_generated.shape[0] == 1:
        depth_img = depth_generated[0]
    else:
        depth_img = depth_generated[0]  # Take first channel if multiple

    output_path = os.path.join(opt.results_dir, opt.name, f'{name}_depth.png')
    Image.fromarray(depth_img).save(output_path)

    if (i + 1) % 10 == 0 or (i + 1) == min(opt.how_many, len(dataset)):
        print(f"  Processed {i + 1}/{min(opt.how_many, len(dataset))} images")

print(f"\n✓ Done! Results saved to: {out_dir}")