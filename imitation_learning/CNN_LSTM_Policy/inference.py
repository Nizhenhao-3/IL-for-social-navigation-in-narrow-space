import torch
import matplotlib.pyplot as plt
from CNN_LSTM import RGBDNetLight  # 你的模型定义
from torch_dataset_all_episode import load_data

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 加载模型
model = RGBDNetLight().to(device)
model.load_state_dict(torch.load("imitation_learning/CNN_LSTM_Policy/checkpoints/rgbdnet_light.pt"))  # 你保存的模型路径
model.eval()

# 加载数据（验证集）
_, val_loader, _ = load_data(
    dataset_dir="hdf5_dataset",  # 数据目录
    num_episodes=45,
    batch_size_train=1,
    batch_size_val=1   # 每次只评估一个episode，便于绘图
)

# 从验证集中取出一条数据
with torch.no_grad():
    for depth, rgb, actions in val_loader:
        depth = depth.to(device).float()  # [B, T, 480, 848]
        rgb = rgb.to(device).float()      # [B, T, 3, 720, 1280]
        actions = actions.to(device).float()  # [B, T, 3]

        preds = model(depth, rgb)  # [B, T, 3]

        # 假设batch_size=1
        preds = preds.squeeze(0).cpu().numpy()     # [T, 3]
        actions = actions.squeeze(0).cpu().numpy() # [T, 3]
        break  # 只用第一批即可

# 可视化
fig, axes = plt.subplots(3, 1, figsize=(12, 8))

labels = ['vx', 'vy', 'vz']
for i in range(3):
    axes[i].plot(preds[:, i], label='Predicted')
    axes[i].plot(actions[:, i], label='Ground Truth')
    axes[i].set_title(f'{labels[i]} over time')
    axes[i].legend()
    axes[i].grid(True)

plt.tight_layout()
plt.show()
