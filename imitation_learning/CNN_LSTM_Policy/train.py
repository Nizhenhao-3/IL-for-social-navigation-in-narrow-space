# from torch_dataset_all_episode import load_data
from torch_dataset_all_episode import load_data
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from CNN_LSTM import RGBDTemporalNetWithSpatialAttention
import numpy as np
import matplotlib.pyplot as plt
dataset_dir='hdf5_dataset'
num_episodes=45

batch_size_train=1
batch_size_val=1
train_dataloader, val_dataloader, norm_stats=load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val)

num_epochs=2000
lr=1e-3
device='cuda'
model=RGBDTemporalNetWithSpatialAttention().cuda()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()



for epoch in range(num_epochs):
    model.train()
    total_loss = 0.0
    num_batches = 0

    # 遍历训练集
    for batch in tqdm(train_dataloader, desc=f"Epoch {epoch+1}"):
        depth, rgb, actions= batch
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
    

# val_losses = []

# for epoch in range(num_epochs):
#     model.train()
#     train_loss = 0.0
#     for depth, rgb, action in train_dataloader:
#         depth, rgb, action = depth.to(device), rgb.to(device), action.to(device)
#         pred = model(depth, rgb)
#         loss = loss_fn(pred, action)
#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()

   
#     train_loss /= len(train_loader)
#     val_loss /= len(val_loader)
#     train_losses.append(train_loss)
#     val_losses.append(val_loss)

#     print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")

#     return train_losses, val_losses


# model = RGBDTemporalNet()
# train_losses, val_losses = train_model(model, train_dataloader, val_loader, num_epochs=10, lr=1e-3)

# # Plot loss
# plt.plot(train_losses, label="Train Loss")
# plt.plot(val_losses, label="Val Loss")
# plt.xlabel("Epoch")
# plt.ylabel("Loss")
# plt.legend()
# plt.title("Training vs Validation Loss")
# plt.grid(True)
# plt.show()