import h5py
data_path="hdf5_dataset/episode_003.hdf5"
# with h5py.File(data_path, 'r') as f:
#     # 读取数据
#     print(f.keys())

#     rgb_data = f['observations/rgb'][()]#[()]
#     # depth_data = f['observations/depth'][()]
#     # twist_data = f['actions/velocity'][()]
#     # print(twist_data)
#     print(rgb_data)



import os
import h5py
import numpy as np
import matplotlib.pyplot as plt

def plot_velocity_from_hdf5(dataset_dir, save_dir, num_episodes=45):
    os.makedirs(save_dir, exist_ok=True)
    norm_stats={'action_mean': np.array([ 0.47497959, -0.07565234,  0.00179626]), 'action_std': np.array([0.11056302, 0.10084624, 0.24042   ])}

    for episode_id in range(1, num_episodes + 1):
        episode_name = f'episode_{str(episode_id).zfill(3)}.hdf5'
        episode_path = os.path.join(dataset_dir, episode_name)

        if not os.path.isfile(episode_path):
            print(f'[WARNING] File not found: {episode_path}')
            continue

        with h5py.File(episode_path, 'r') as f:
            # 尝试读取 velocity
            if '/actions/velocity' not in f:
                print(f'[WARNING] velocity not found in: {episode_path}')
                continue

            velocity = f['/actions/velocity'][()]  # shape: (N, 3)
            ## 进行归一化
            # velocity = (velocity - norm_stats["action_mean"]) / norm_stats["action_std"]
            num_frames = np.arange(velocity.shape[0])

            plt.figure(figsize=(10, 6))
            plt.plot(num_frames, velocity[:, 0], label='Linear Velocity X', color='b')
            plt.plot(num_frames, velocity[:, 1], label='Linear Velocity Y', color='g')
            plt.plot(num_frames, velocity[:, 2], label='Angular Velocity Z', color='r')

            plt.xlabel('Frame Index')
            plt.ylabel('Velocity')
            plt.title(f'Velocity Over Time - Episode {episode_id:03d}')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()

            save_path = os.path.join(save_dir, f'episode_{episode_id:03d}_velocity.png')
            plt.savefig(save_path)
            plt.close()
            print(f'Saved: {save_path}')

# 示例调用
# plot_velocity_from_hdf5('hdf5_dataset', 'velocity_imgs/normalize_velocity_imgs', num_episodes=45)
