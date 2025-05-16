from imitation_learning.CNN_LSTM_Policy.torch_dataset_all_episode import load_data
# from imitation_learning.torch_dataset import load_data
import numpy as np
dataset_dir='hdf5_dataset'
num_episodes=45

batch_size_train=1
batch_size_val=1
train_dataloader, val_dataloader, norm_stats=load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val)

for batch in train_dataloader:
    depth_img_data, rgb_img_data, action_data= batch
    print(depth_img_data.shape)
    print(rgb_img_data.shape)
    print(action_data.shape)
    
    
    # print(is_pad.shape)
