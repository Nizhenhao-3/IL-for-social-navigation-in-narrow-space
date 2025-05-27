
import argparse
import torch
import h5py
from detr.models.backbone import build_backbone

from torch_dataset_all_episode import load_data

dataset_dir = 'hdf5_dataset'
num_episodes = 45
batch_size_val = 1
_, val_dataloader, norm_stats = load_data(dataset_dir, num_episodes, batch_size_train=1, batch_size_val=batch_size_val)

for i, (depth, rgb, actions_gt) in enumerate(val_dataloader):
    print(f"Batch {i+1}:")
    print(f"  Depth shape: {depth.shape}, dtype: {depth.dtype}")
    print(f"  RGB shape: {rgb.shape}, dtype: {rgb.dtype}")
    print(f"  Actions GT shape: {actions_gt.shape}, dtype: {actions_gt.dtype}")
