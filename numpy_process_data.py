import cv2
import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt

import h5py

png_dataset_path = 'png_dataset'
for i in range(1,46):

    idx= str(i).zfill(3)
    episode_id = f'episode_{idx}'
    print(f"episode_id: {episode_id}")
    episode_path=os.path.join(png_dataset_path,episode_id)
    rgb_dir = os.path.join(episode_path, 'rgb')
    depth_dir = os.path.join(episode_path, 'depth')
    output_csv_path = os.path.join(episode_path, 'labels.csv')

#处理深度图
    depth_images = []
    
    for j in range(380):
        idx=str(j).zfill(4)
        depth_img_name=os.path.join(depth_dir,f'{idx}_depth.png')
        depth_image = cv2.imread(depth_img_name, cv2.IMREAD_UNCHANGED)
        # print(depth_image.shape)
        depth_images.append(depth_image)

    all_depth_images = np.stack(depth_images,axis=0)
    # print(all_depth_images.shape)

#处理RGB图
    rgb_images = []
    
    for k in range(380):
        idx=str(k).zfill(4)
        rgb_img_name=os.path.join(rgb_dir,f'{idx}_rgb.png')
        rgb_image = cv2.imread(rgb_img_name, cv2.IMREAD_UNCHANGED)
        # print(depth_image.shape)
        rgb_images.append(rgb_image)
    all_rgb_images = np.stack(rgb_images,axis=0)
    

     # 读取 CSV 文件
    df = pd.read_csv(output_csv_path)
    features = df[['linear_velocity_x', 'linear_velocity_y', 'angular_velocity']].to_numpy()

    # 输出路径
    path = 'hdf5_dataset'
    hdf5_path = os.path.join(path, f'{episode_id}.hdf5')
    print(f'输出路径: {hdf5_path}')

    # 创建文件
    with h5py.File(hdf5_path, 'w') as f:
        grp = f.create_group('observations')
        grp.create_dataset('rgb', data=all_rgb_images)
        grp.create_dataset('depth', data=all_depth_images)
        grp2=f.create_group('actions')
        grp2.create_dataset('velocity', data=features)

    print(f'✅ HDF5 文件已保存：{hdf5_path}')
    














# # print(features)

# # num_frames = range(features.shape[0])
# # plt.figure(figsize=(10, 6))
# # plt.plot(num_frames, features[:, 0], label='Linear Velocity X', color='b')
# # plt.plot(num_frames, features[:, 1], label='Linear Velocity Y', color='g')
# # plt.plot(num_frames, features[:, 2], label='Angular Velocity Z', color='r')

# # plt.xlabel('Frame Index')
# # plt.ylabel('Velocity')
# # plt.title('Velocity Over Time')
# # plt.legend()
# # plt.grid(True)
# # plt.tight_layout()
# # plt.show()


