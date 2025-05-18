# from torch_dataset_all_episode import load_data
from torch_dataset_all_episode import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
# from CNN_LSTM import RGBDTemporalNetWithSpatialAttention
from CNN_LSTM import RGBDNetLight
import numpy as np
import matplotlib.pyplot as plt
import os
dataset_dir='hdf5_dataset'
num_episodes=45

batch_size_train=1
batch_size_val=1
train_dataloader, val_dataloader, norm_stats=load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val)

lr=1e-3
device='cuda'
model=RGBDNetLight().cuda()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()


save_dir='imitation_learning/CNN_LSTM_Policy/checkpoints'



num_epochs = 200
save_path = os.path.join(save_dir, 'rgbdnet_light.pt')

num_additional_epochs = 200

# 如果模型文件不存在，从头开始训练；如果存在，加载继续训练
if not os.path.exists(save_path):
    print("从头开始训练...")
    start_epoch = 0
else:
    print("加载已有模型继续训练...")
    model.load_state_dict(torch.load(save_path))
    start_epoch = 200  # 假设之前已经训练了200轮

for epoch in range(start_epoch, start_epoch + num_additional_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 遍历训练集
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):

        
        depth, rgb, actions= batch
        
        actions = actions.float()

        depth = depth.to(device)
        rgb = rgb.to(device)
        actions = actions.to(device)
        # depth = batch['depth'].cuda()     # [B, T, 1, 480, 848]
        # rgb = batch['rgb'].cuda()         # [B, T, 3, 720, 1280]
        # actions = batch['actions'].cuda() # [B, T, 3]

        # 前向传播
        preds = model(depth, rgb)         # [B, T, 3]

        # 计算损失
        loss = criterion(preds, actions)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches
    print(f"[Epoch {epoch+1}] Train Loss: {avg_loss:.4f}")

save_path = os.path.join(save_dir, 'rgbdnet_light.pt')
torch.save(model.state_dict(), save_path)
print("模型已保存为 saved_model.pt")

