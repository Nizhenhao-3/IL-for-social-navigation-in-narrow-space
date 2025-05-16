import numpy as np
import torch
import os
import h5py
from torch.utils.data import TensorDataset, DataLoader

import IPython
e = IPython.embed

dataset_dir="hdf5_dataset"



class EpisodicDataset(torch.utils.data.Dataset):
    def __init__(self, episode_ids, dataset_dir, norm_stats):
        super().__init__()
        self.episode_ids = episode_ids#列表，随机生成的episode id
        self.dataset_dir = dataset_dir
      
        self.norm_stats = norm_stats#标准化所需的统计量
        #在初始化时，主动调用一次 __getitem__(0)，即尝试读取第一个 episode 的数据；
        self.__getitem__(0) # initialize self.is_sim

    def __len__(self):
        return len(self.episode_ids)

    def __getitem__(self, index):
        episode_id = str(self.episode_ids[index]).zfill(3)
        dataset_path = os.path.join(self.dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            episode_len = root['actions/velocity'].shape[0]
            
            # print(f"episode_id: {episode_id}, start_ts: {start_ts}")

            # 图像
            depth_img = root['/observations/depth'][()]
            rgb_img = root['/observations/rgb'][()]

            # 当前帧动作
            action = root['actions/velocity'][()]  # shape: 

        # Tensor 转换
        depth_img_data = torch.from_numpy(depth_img)/ 255.0
        rgb_img_data = torch.from_numpy(rgb_img) / 255.0
        rgb_img_data=torch.einsum('t h w c -> t c h w', rgb_img_data)

        action_data = torch.from_numpy(action).float()
        action_data = (action_data - self.norm_stats["action_mean"]) / self.norm_stats["action_std"]

        return depth_img_data, rgb_img_data, action_data

        
        
def get_norm_stats(dataset_dir, num_episodes):
    #num_episode=45
    
    all_action_data = []
    for episode_idx in range(1,num_episodes+1):
        episode_id = str(episode_idx).zfill(3)
        dataset_path = os.path.join(dataset_dir, f'episode_{episode_id}.hdf5')
        with h5py.File(dataset_path, 'r') as root:
            
            action = root['actions/velocity'][()]
        
        all_action_data.append(torch.from_numpy(action))
    
    all_action_data = torch.stack(all_action_data)#torch.stack将维度相同的tensor拼接在一起，形成一个新的tensor
    all_action_data = all_action_data

    # normalize action data
    action_mean = all_action_data.mean(dim=[0, 1], keepdim=True)#在所有episode和所有时间步上计算均值
    action_std = all_action_data.std(dim=[0, 1], keepdim=True)
    action_std = torch.clip(action_std, 1e-2, np.inf) # clipping

    # normalize qpos data
    

    stats = {"action_mean": action_mean.numpy().squeeze(), "action_std": action_std.numpy().squeeze(),
             }

    return stats


def load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val):
    print(f'\nData from: {dataset_dir}\n')
    # obtain train test split
    train_ratio = 0.8#百分之80用于训练
    shuffled_indices = np.random.permutation(num_episodes)+1#打乱
    train_indices = shuffled_indices[:int(train_ratio * num_episodes)]
    val_indices = shuffled_indices[int(train_ratio * num_episodes):]

    # obtain normalization stats for qpos and action
    norm_stats = get_norm_stats(dataset_dir, num_episodes)

    # construct dataset and dataloader
    train_dataset = EpisodicDataset(train_indices, dataset_dir,  norm_stats)
    val_dataset = EpisodicDataset(val_indices, dataset_dir,  norm_stats)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size_train, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=True, pin_memory=True, num_workers=1, prefetch_factor=1)

    return train_dataloader, val_dataloader, norm_stats
