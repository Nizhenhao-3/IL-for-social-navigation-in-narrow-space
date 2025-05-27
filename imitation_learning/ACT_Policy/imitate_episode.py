import torch
import numpy as np
import os
import pickle
import argparse
import matplotlib.pyplot as plt
from copy import deepcopy
from tqdm import tqdm
from einops import rearrange

# from constants import DT
# from constants import PUPPET_GRIPPER_JOINT_OPEN
from torch_dataset import load_data # data functions

from torch_dataset import compute_dict_mean, set_seed, detach_dict # helper functions

#TO do
from policy import ACTPolicy, CNNMLPPolicy

from torch_dataset_all_episode import load_data_all


import IPython
e = IPython.embed

def main(args):
    set_seed(1)
    # command line parameters
    is_eval = args['eval']
    ckpt_dir = args['ckpt_dir']
    policy_class = args['policy_class']

    
    batch_size_train = args['batch_size']
    batch_size_val = args['batch_size']
    num_epochs = args['num_epochs']

    # get task parameters
    dataset_dir='hdf5_dataset'
    num_episodes = 45
    episode_len = 380

   
    state_dim = 3# output dim
    lr_backbone = 1e-5
    backbone = 'resnet18'
    if policy_class == 'ACT':
        enc_layers = 4
        dec_layers = 7
        nheads = 8
        policy_config = {'lr': args['lr'],
                         'num_queries': args['chunk_size'],
                         'kl_weight': args['kl_weight'],
                         'hidden_dim': args['hidden_dim'],
                         'dim_feedforward': args['dim_feedforward'],
                         'lr_backbone': lr_backbone,
                         'backbone': backbone,
                         'enc_layers': enc_layers,
                         'dec_layers': dec_layers,
                         'nheads': nheads,
                        #  'camera_names': camera_names,
                         }
    elif policy_class == 'CNNMLP':
        policy_config = {'lr': args['lr'], 'lr_backbone': lr_backbone, 'backbone' : backbone, 'num_queries': 1,
                         }
    else:
        raise NotImplementedError

    config = {
        'num_epochs': num_epochs,
        'ckpt_dir': ckpt_dir,
        'episode_len': episode_len,
        'state_dim': state_dim,
        'lr': args['lr'],
        'policy_class': policy_class,
        
        'policy_config': policy_config,
        
        'seed': args['seed'],
        'temporal_agg': args['temporal_agg'],
        
    }
#TODO
    if is_eval:
        ckpt_names = 'policy_best.ckpt'
        eval_bc(config, ckpt_names, save_episode=True)
        
        exit()




#load data
    train_dataloader, val_dataloader, stats = load_data(dataset_dir, num_episodes,  batch_size_train, batch_size_val)


    # save dataset stats
    if not os.path.isdir(ckpt_dir):
        os.makedirs(ckpt_dir)
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'wb') as f:
        pickle.dump(stats, f)

    best_ckpt_info = train_bc(train_dataloader, val_dataloader, config)
    best_epoch, min_val_loss, best_state_dict = best_ckpt_info

    # save best checkpoint
    ckpt_path = os.path.join(ckpt_dir, f'policy_best.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Best ckpt, val loss {min_val_loss:.6f} @ epoch{best_epoch}')


def make_policy(policy_class, policy_config):
    if policy_class == 'ACT':
        policy = ACTPolicy(policy_config)
    elif policy_class == 'CNNMLP':
        policy = CNNMLPPolicy(policy_config)
    else:
        raise NotImplementedError
    return policy


def make_optimizer(policy_class, policy):
    if policy_class == 'ACT':
        optimizer = policy.configure_optimizers()
    elif policy_class == 'CNNMLP':
        optimizer = policy.configure_optimizers()
    else:
        raise NotImplementedError
    return optimizer


def get_image(ts):
    #获取一个时间步(ts)的图像数据
    curr_image = rearrange(ts.observation['rgb'], 'h w c -> c h w')
    
    # curr_image = np.stack(curr_images, axis=0)
    curr_image = torch.from_numpy(curr_image / 255.0).float().cuda().unsqueeze(0)
    return curr_image





def eval_bc(config, ckpt_name, save_episode=True):
    set_seed(1000)
    ckpt_dir = config['ckpt_dir']
    state_dim = config['state_dim']
    
    policy_class = config['policy_class']
    
    policy_config = config['policy_config']
    
    max_timesteps = config['episode_len']
    
    temporal_agg = config['temporal_agg']

    dataset_dir = 'hdf5_dataset'
    num_episodes = 45
    batch_size_val = 1
    _, val_dataloader, norm_stats = load_data_all(dataset_dir, num_episodes, batch_size_train=1, batch_size_val=batch_size_val)
        

    # load policy and stats
    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    policy = make_policy(policy_class, policy_config)
    loading_status = policy.load_state_dict(torch.load(ckpt_path))
    print(loading_status)

    policy.cuda()
    policy.eval()
    print(f'Loaded: {ckpt_path}')
    stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
    with open(stats_path, 'rb') as f:
        stats = pickle.load(f)


    pre_process = lambda action: (action - stats['action_mean']) / stats['action_std']
    post_process = lambda a: a * stats['action_std'] + stats['action_mean']

 



    query_frequency = policy_config['num_queries']
    if temporal_agg:
        query_frequency = 1
        num_queries = policy_config['num_queries']


    

    max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks


    ### evaluation loop
    if temporal_agg:
        all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()


    image_list = [] # for visualization
    depth_img_list = [] # for visualization


    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    

    pred_actions_all = []
    gt_actions_all = []

    with torch.inference_mode():
        for i, (depth, rgb, actions_gt) in enumerate(val_dataloader):
            if i > 0:  # 只推理第一集
                break

            for t in range(max_timesteps):
                rgb_frame = rgb[:, t].to(device)       # [1, 3, H, W]
                depth_frame = depth[:, t].to(device)   # [1,  H, W]
                

                ### process previous timestep to get qpos and image_list
               
                image_list.append(rgb_frame)
                depth_img_list.append(depth_frame)

                curr_image=rgb_frame
                

                ### query policy
                if config['policy_class'] == "ACT":
                    if t % query_frequency == 0:
                        all_actions = policy(depth_frame, curr_image)#[num_queries, bs,state_dim]
                        all_actions=all_actions.squeeze(1)#[num_queries, state_dim]
                    if temporal_agg:
                        all_time_actions[[t], t:t+num_queries] = all_actions
                        actions_for_curr_step = all_time_actions[:, t]
                        actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
                        actions_for_curr_step = actions_for_curr_step[actions_populated]
                        k = 0.01
                        exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
                        exp_weights = exp_weights / exp_weights.sum()
                        exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
                        raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
                    else:
                        raw_action = all_actions[:, t % query_frequency]
                
                else:
                    raise NotImplementedError

                ### post-process actions
                raw_action = raw_action.squeeze(0).cpu().numpy()
                action = post_process(raw_action)
                pred_actions_all.append(action)

                gt_unscaled = actions_gt[:, t] * torch.from_numpy(norm_stats['action_std']) + torch.from_numpy(norm_stats['action_mean'])
                gt_actions_all.append(gt_unscaled.numpy().squeeze())

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



    return


# def eval_bc(config, ckpt_name, save_episode=True):
#     set_seed(1000)
#     ckpt_dir = config['ckpt_dir']
#     state_dim = config['state_dim']
    
#     policy_class = config['policy_class']
    
#     policy_config = config['policy_config']
    
#     max_timesteps = config['episode_len']
    
#     temporal_agg = config['temporal_agg']
    

#     # load policy and stats
#     ckpt_path = os.path.join(ckpt_dir, ckpt_name)
#     policy = make_policy(policy_class, policy_config)
#     loading_status = policy.load_state_dict(torch.load(ckpt_path))
#     print(loading_status)

#     policy.cuda()
#     policy.eval()
#     print(f'Loaded: {ckpt_path}')
#     stats_path = os.path.join(ckpt_dir, f'dataset_stats.pkl')
#     with open(stats_path, 'rb') as f:
#         stats = pickle.load(f)


#     pre_process = lambda action: (action - stats['action_mean']) / stats['action_std']
#     post_process = lambda a: a * stats['action_std'] + stats['action_mean']

 

#     query_frequency = policy_config['num_queries']
#     if temporal_agg:
#         query_frequency = 1
#         num_queries = policy_config['num_queries']

#     max_timesteps = int(max_timesteps * 1) # may increase for real-world tasks

#     num_rollouts = 50
#     episode_returns = []
#     highest_rewards = []
#     for rollout_id in range(num_rollouts):
#         rollout_id += 0
#         ### set task
        

#         ### evaluation loop
#         if temporal_agg:
#             all_time_actions = torch.zeros([max_timesteps, max_timesteps+num_queries, state_dim]).cuda()

        

#         qpos_history = torch.zeros((1, max_timesteps, state_dim)).cuda()
#         image_list = [] # for visualization
#         qpos_list = []
#         target_qpos_list = []
#         rewards = []


#         with torch.inference_mode():
#             for t in range(max_timesteps):
#                 ### update onscreen render and wait for DT
               

#                 ### process previous timestep to get qpos and image_list
#                 obs = ts.observation
#                 if 'images' in obs:
#                     image_list.append(obs['images'])
#                 else:
#                     image_list.append({'main': obs['image']})
#                 qpos_numpy = np.array(obs['qpos'])
#                 qpos = pre_process(qpos_numpy)
#                 qpos = torch.from_numpy(qpos).float().cuda().unsqueeze(0)
#                 qpos_history[:, t] = qpos
#                 curr_image = get_image(ts, camera_names)

#                 ### query policy
#                 if config['policy_class'] == "ACT":
#                     if t % query_frequency == 0:
#                         all_actions = policy(qpos, curr_image)
#                     if temporal_agg:
#                         all_time_actions[[t], t:t+num_queries] = all_actions
#                         actions_for_curr_step = all_time_actions[:, t]
#                         actions_populated = torch.all(actions_for_curr_step != 0, axis=1)
#                         actions_for_curr_step = actions_for_curr_step[actions_populated]
#                         k = 0.01
#                         exp_weights = np.exp(-k * np.arange(len(actions_for_curr_step)))
#                         exp_weights = exp_weights / exp_weights.sum()
#                         exp_weights = torch.from_numpy(exp_weights).cuda().unsqueeze(dim=1)
#                         raw_action = (actions_for_curr_step * exp_weights).sum(dim=0, keepdim=True)
#                     else:
#                         raw_action = all_actions[:, t % query_frequency]
#                 elif config['policy_class'] == "CNNMLP":
#                     raw_action = policy(qpos, curr_image)
#                 else:
#                     raise NotImplementedError

#                 ### post-process actions
#                 raw_action = raw_action.squeeze(0).cpu().numpy()
#                 action = post_process(raw_action)
#                 target_qpos = action

#                 ### step the environment
#                 ts = env.step(target_qpos)

#                 ### for visualization
#                 qpos_list.append(qpos_numpy)
#                 target_qpos_list.append(target_qpos)
#                 rewards.append(ts.reward)

#             plt.close()
#         # if real_robot:
#         #     move_grippers([env.puppet_bot_left, env.puppet_bot_right], [PUPPET_GRIPPER_JOINT_OPEN] * 2, move_time=0.5)  # open
#         #     pass

#         rewards = np.array(rewards)
#         episode_return = np.sum(rewards[rewards!=None])
#         episode_returns.append(episode_return)
#         episode_highest_reward = np.max(rewards)
#         highest_rewards.append(episode_highest_reward)
#         print(f'Rollout {rollout_id}\n{episode_return=}, {episode_highest_reward=}, {env_max_reward=}, Success: {episode_highest_reward==env_max_reward}')

#         # if save_episode:
#         #     save_videos(image_list, DT, video_path=os.path.join(ckpt_dir, f'video{rollout_id}.mp4'))

#     success_rate = np.mean(np.array(highest_rewards) == env_max_reward)
#     avg_return = np.mean(episode_returns)
#     summary_str = f'\nSuccess rate: {success_rate}\nAverage return: {avg_return}\n\n'
#     for r in range(env_max_reward+1):
#         more_or_equal_r = (np.array(highest_rewards) >= r).sum()
#         more_or_equal_r_rate = more_or_equal_r / num_rollouts
#         summary_str += f'Reward >= {r}: {more_or_equal_r}/{num_rollouts} = {more_or_equal_r_rate*100}%\n'

#     print(summary_str)

#     # save success rate to txt
#     result_file_name = 'result_' + ckpt_name.split('.')[0] + '.txt'
#     with open(os.path.join(ckpt_dir, result_file_name), 'w') as f:
#         f.write(summary_str)
#         f.write(repr(episode_returns))
#         f.write('\n\n')
#         f.write(repr(highest_rewards))

#     return success_rate, avg_return


def forward_pass(data, policy):
    depth_img_data, rgb_img_data, action_data, is_pad=data
    depth_img_data,rgb_img_data, action_data, is_pad = depth_img_data.cuda(), rgb_img_data.cuda(), action_data.cuda(), is_pad.cuda()
   
    return policy(depth_img_data, rgb_img_data, action_data, is_pad) # TODO remove None


def train_bc(train_dataloader, val_dataloader, config):
    num_epochs = config['num_epochs']
    ckpt_dir = config['ckpt_dir']
    seed = config['seed']
    policy_class = config['policy_class']
    policy_config = config['policy_config']

    set_seed(seed)

    policy = make_policy(policy_class, policy_config)
    policy.cuda()
    optimizer = make_optimizer(policy_class, policy)

    train_history = []
    validation_history = []
    min_val_loss = np.inf
    best_ckpt_info = None
    for epoch in tqdm(range(num_epochs)):
        print(f'\nEpoch {epoch}')
        # validation
        with torch.inference_mode():
            policy.eval()
            epoch_dicts = []
            for batch_idx, data in enumerate(val_dataloader):
                forward_dict = forward_pass(data, policy)
                epoch_dicts.append(forward_dict)
            epoch_summary = compute_dict_mean(epoch_dicts)
            validation_history.append(epoch_summary)

            epoch_val_loss = epoch_summary['loss']
            if epoch_val_loss < min_val_loss:
                min_val_loss = epoch_val_loss
                best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))
        print(f'Val loss:   {epoch_val_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        # training
        policy.train()
        optimizer.zero_grad()
        for batch_idx, data in enumerate(train_dataloader):
            forward_dict = forward_pass(data, policy)
            # backward
            loss = forward_dict['loss']
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            train_history.append(detach_dict(forward_dict))
        epoch_summary = compute_dict_mean(train_history[(batch_idx+1)*epoch:(batch_idx+1)*(epoch+1)])
        epoch_train_loss = epoch_summary['loss']
        print(f'Train loss: {epoch_train_loss:.5f}')
        summary_string = ''
        for k, v in epoch_summary.items():
            summary_string += f'{k}: {v.item():.3f} '
        print(summary_string)

        if epoch % 100 == 0:
            ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{epoch}_seed_{seed}.ckpt')
            torch.save(policy.state_dict(), ckpt_path)
            plot_history(train_history, validation_history, epoch, ckpt_dir, seed)

    ckpt_path = os.path.join(ckpt_dir, f'policy_last.ckpt')
    torch.save(policy.state_dict(), ckpt_path)

    best_epoch, min_val_loss, best_state_dict = best_ckpt_info
    ckpt_path = os.path.join(ckpt_dir, f'policy_epoch_{best_epoch}_seed_{seed}.ckpt')
    torch.save(best_state_dict, ckpt_path)
    print(f'Training finished:\nSeed {seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}')

    # save training curves
    plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed)

    return best_ckpt_info


def plot_history(train_history, validation_history, num_epochs, ckpt_dir, seed):
    # save training curves
    for key in train_history[0]:
        plot_path = os.path.join(ckpt_dir, f'train_val_{key}_seed_{seed}.png')
        plt.figure()
        train_values = [summary[key].item() for summary in train_history]
        val_values = [summary[key].item() for summary in validation_history]
        plt.plot(np.linspace(0, num_epochs-1, len(train_history)), train_values, label='train')
        plt.plot(np.linspace(0, num_epochs-1, len(validation_history)), val_values, label='validation')
        # plt.ylim([-0.1, 1])
        plt.tight_layout()
        plt.legend()
        plt.title(key)
        plt.savefig(plot_path)
    print(f'Saved plots to {ckpt_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--onscreen_render', action='store_true')
    parser.add_argument('--ckpt_dir', action='store', type=str, help='ckpt_dir', required=True)
    parser.add_argument('--policy_class', action='store', type=str, help='policy_class, capitalize', required=True)
    # parser.add_argument('--task_name', action='store', type=str, help='task_name', required=True)
    parser.add_argument('--batch_size', action='store', type=int, help='batch_size', required=True)
    parser.add_argument('--seed', action='store', type=int, help='seed', required=True)
    parser.add_argument('--num_epochs', action='store', type=int, help='num_epochs', required=True)
    parser.add_argument('--lr', action='store', type=float, help='lr', required=True)

# for ACT
    parser.add_argument('--kl_weight', action='store', type=int, help='KL Weight', required=False)
    parser.add_argument('--chunk_size', action='store', type=int, help='chunk_size', required=False)
    parser.add_argument('--hidden_dim', action='store', type=int, help='hidden_dim', required=False)
    parser.add_argument('--dim_feedforward', action='store', type=int, help='dim_feedforward', required=False)
    parser.add_argument('--temporal_agg', action='store_true')
    
    main(vars(parser.parse_args()))