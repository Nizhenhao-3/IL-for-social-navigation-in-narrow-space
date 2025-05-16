from torch_dataset_all_episode import load_data
from torch_dataset_all_episode import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from CNN_LSTM import RGBDTemporalNetWithSpatialAttention
import numpy as np
dataset_dir='hdf5_dataset'
num_episodes=45

batch_size_train=1
batch_size_val=1
train_dataloader, val_dataloader, norm_stats=load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val)

device='cuda'

for batch in train_dataloader:
        depth, rgb, actions= batch
        depth = depth.to(device)
        rgb = rgb.to(device)
        actions = actions.to(device).float()
        print(depth.dtype)
        print(rgb.dtype)
        print(actions.dtype)