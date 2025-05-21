# from torch_dataset_all_episode import load_data
# from torch_dataset_all_episode import load_data
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from tqdm import tqdm
# # from CNN_LSTM import RGBDTemporalNetWithSpatialAttention
# import numpy as np
# dataset_dir='hdf5_dataset'
# num_episodes=45

# batch_size_train=1
# batch_size_val=1
# train_dataloader, val_dataloader, norm_stats=load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val)

# device='cuda'

# for batch in train_dataloader:
#         depth, rgb, actions= batch
#         depth = depth.to(device)
#         rgb = rgb.to(device)
#         actions = actions.to(device).float()
#         print(depth.dtype)
#         print(rgb.dtype)
#         print(actions.dtype)



import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from CNN_LSTM import RGBDNetLight  # 模型定义
from torch_dataset_all_episode import load_data     # 数据加载和归一化
import torch.nn.functional as F

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# 加载模型
model = RGBDNetLight().to(device)
model_path = 'imitation_learning/CNN_LSTM_Policy/checkpoints/rgbdnet_light.pt'
model.load_state_dict(torch.load(model_path))
model.eval()

# 加载数据（batch_size = 1 即逐帧推理）
dataset_dir = 'hdf5_dataset'
num_episodes = 45
batch_size_val = 1
_, val_dataloader, norm_stats = load_data(dataset_dir, num_episodes, batch_size_train=1, batch_size_val=batch_size_val)

# 保存对比结果
pred_actions_all = []
gt_actions_all = []

# 取一集推理验证
with torch.no_grad():
    for i, (depth, rgb, actions_gt) in enumerate(val_dataloader):
        if i > 0:  # 只推理第一集
            break

        # 初始化 LSTM 状态
        hidden_state = None
        depth = depth.unsqueeze(2)  # 变成 [B, T, 1, 480, 848]

        B, T, C, H, W = rgb.shape

        for t in range(T):
            rgb_frame = rgb[:, t].to(device)       # [1, 3, H, W]
            depth_frame = depth[:, t].to(device)   # [1, 1, H, W]

            # 编码
            rgb_feat = model.rgb_encoder(rgb_frame).flatten(1)   # [1, 256]
            depth_feat = model.depth_encoder(depth_frame).flatten(1)  # [1, 128]
            fused = torch.cat([rgb_feat, depth_feat], dim=1).unsqueeze(1)  # [1, 1, 384]

            # LSTM + MLP
            lstm_out, hidden_state = model.lstm(fused, hidden_state)  # [1, 1, 64]
            pred = model.mlp(lstm_out.squeeze(1))                     # [1, 3]

            # 反标准化
            pred_unscaled = pred.cpu() * torch.from_numpy(norm_stats['action_std']) + torch.from_numpy(norm_stats['action_mean'])
            gt_unscaled = actions_gt[:, t] * torch.from_numpy(norm_stats['action_std']) + torch.from_numpy(norm_stats['action_mean'])

            pred_actions_all.append(pred_unscaled.numpy().squeeze())
            gt_actions_all.append(gt_unscaled.numpy().squeeze())

# 转为数组
pred_actions_all = np.array(pred_actions_all)  # [T, 3]
gt_actions_all = np.array(gt_actions_all)      # [T, 3]

# 绘图对比
plt.figure(figsize=(12, 6))
labels = ['x velocity', 'y velocity', 'z velocity']
for i in range(3):
    plt.subplot(3, 1, i+1)
    plt.plot(gt_actions_all[:, i], label='Ground Truth', color='black')
    plt.plot(pred_actions_all[:, i], label='Predicted', linestyle='--')
    plt.ylabel(labels[i])
    if i == 0:
        plt.title('Predicted vs Ground Truth Velocities')
    if i == 2:
        plt.xlabel('Time step')
    plt.legend()

plt.tight_layout()
plt.show()
