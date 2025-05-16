

import os
import csv
import numpy as np
import cv2
import rosbag2_py
from rclpy.serialization import deserialize_message
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

sample_dataset_path = 'sample_without_lidar'
    
for i in range(1,46):
    episode_id = str(i).zfill(3)
    # print(f"episode_id: {episode_id}")
    episode_path=os.path.join(sample_dataset_path,f'sample_{episode_id}')
    bag_file=os.path.join(episode_path,f'sample_{episode_id}_0.db3')
    rgb_dir = os.path.join('png_dataset', f'episode_{episode_id}', 'rgb')
    depth_dir = os.path.join('png_dataset', f'episode_{episode_id}', 'depth')
    output_csv_path = os.path.join('png_dataset', f'episode_{episode_id}', 'labels.csv')

    
    os.makedirs(rgb_dir, exist_ok=True)
    os.makedirs(depth_dir, exist_ok=True)

    # 初始化
    bridge = CvBridge()
    frame_count_rgb = 0
    frame_count_depth = 0
    frame_count_twist = 0
    max_frames = 380

    # 数据缓存列表
    data_rows = []
    current_twist = None
    last_depth_filename = None

    # === 打开 rosbag ===
    bag = rosbag2_py.SequentialReader()
    storage_options = rosbag2_py.StorageOptions(uri=bag_file, storage_id='sqlite3')
    converter_options = rosbag2_py.ConverterOptions(input_serialization_format='cdr', output_serialization_format='cdr')
    bag.open(storage_options, converter_options)

    while bag.has_next():
        topic, msg_bytes, t = bag.read_next()

        if topic == '/odom_synced' and frame_count_twist < max_frames:
            msg = deserialize_message(msg_bytes, Odometry)
            current_twist = (
                round(msg.twist.twist.linear.x, 3),
                round(msg.twist.twist.linear.y, 3),
                round(msg.twist.twist.angular.z, 3)
            )
            frame_count_twist += 1

        elif topic == '/depth_img_synced' and frame_count_depth < max_frames:
            msg = deserialize_message(msg_bytes, Image)
            depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding='32FC1')

            # 归一化为 0~255 并转换为 uint8
            depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
            depth_image = depth_image.astype(np.uint8)

            depth_filename = f'{frame_count_depth:04d}_depth.png'
            depth_path = os.path.join(depth_dir, depth_filename)
            cv2.imwrite(depth_path, depth_image)
            last_depth_filename = depth_filename
            frame_count_depth += 1

        elif topic == '/img_synced' and current_twist is not None and frame_count_rgb < max_frames:
            msg = deserialize_message(msg_bytes, Image)
            rgb_image = bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')

            rgb_filename = f'{frame_count_rgb:04d}_rgb.png'
            rgb_path = os.path.join(rgb_dir, rgb_filename)
            cv2.imwrite(rgb_path, rgb_image)
            frame_count_rgb += 1

            # 保存到缓存中（只在depth图像也至少处理了一帧时记录）
            data_rows.append({
                'rgb_image_filename': rgb_filename,
                'depth_image_filename': last_depth_filename if last_depth_filename else '',
                'linear_velocity_x': current_twist[0],
                'linear_velocity_y': current_twist[1],
                'angular_velocity': current_twist[2]
            })

        # 三个话题都超过 max_frames 就结束
        if (frame_count_rgb >= max_frames and 
            frame_count_depth >= max_frames and 
            frame_count_twist >= max_frames):
            break

    # === 写入 CSV 文件 ===
    with open(output_csv_path, 'w', newline='') as csvfile:
        fieldnames = ['rgb_image_filename', "depth_image_filename", 'linear_velocity_x', "linear_velocity_y", 'angular_velocity']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in data_rows:
            writer.writerow(row)

    print(f"✅ 提取完成：")
    print(f"   - RGB 图像数量: {frame_count_rgb}")
    print(f"   - 深度图像数量: {frame_count_depth}")
    print(f"   - Twist 信息数量: {frame_count_twist}")
    print(f"📄 CSV 文件已保存至 {output_csv_path}")


