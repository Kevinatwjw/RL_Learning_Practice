import numpy as np
import random
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from gym.utils.env_checker import check_env
from Policygradient_baseline_class import A2C_target
from gym.wrappers import TimeLimit

if __name__ == "__main__":
    def moving_average(a, window_size):
        ''' 生成序列 a 的滑动平均序列 '''
        cumulative_sum = np.cumsum(np.insert(a, 0, 0)) 
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        r = np.arange(1, window_size-1, 2)
        begin = np.cumsum(a[:window_size-1])[::2] / r
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
        return np.concatenate((begin, middle, end))

    def set_seed(env, seed=50):
        ''' 设置随机种子 '''
        env.action_space.seed(seed)
        env.reset(seed=seed)
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    state_dim = 4               # 环境观测维度
    action_range = 2            # 环境动作空间大小
    actor_lr = 1e-3
    critic_lr = 1e-2
    target_weight = 0.95
    num_episodes = 1000
    hidden_dim = 64
    gamma = 0.98
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # build environment
    env_name = 'CartPole-v1'
    env = gym.make(env_name, render_mode='rgb_array')
    check_env(env.unwrapped)    # 检查环境是否符合 gym 规范
    set_seed(env, 0)

    # build agent
    agent = A2C_target(state_dim, hidden_dim, action_range, actor_lr, critic_lr, gamma, target_weight,device)

    # start training
    return_list = []
    for i in range(20):
        with tqdm(total=int(num_episodes / 20), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 20)):
                episode_return = 0
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                state, _ = env.reset()

                # 以当前策略交互得到一条轨迹
                while True:
                    action = agent.take_action(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(terminated or truncated)
                    state = next_state
                    episode_return += reward
                    
                    if terminated or truncated:
                        env.render()
                        break
                    #env.render()

                # 用当前策略收集的数据进行 on-policy 更新
                agent.update(transition_dict)

                # 更新进度条
                return_list.append(episode_return)
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % episode_return,
                    'ave return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(1)

    # 训练结束后关闭环境
    env.close()
    
    # 创建结果目录，用于保存训练结果图
    # exist_ok=True：如果目录已存在，不会报错
    os.makedirs('./CartPole/result', exist_ok=True)
    
    # 显示策略性能
    # 使用滑动平均平滑回报曲线，窗口大小为 29
    mv_return_list = moving_average(return_list, 29)
    # 生成回合编号列表 [0, 1, ..., len(return_list)-1]
    episodes_list = list(range(len(return_list)))
    # 创建画布，设置大小为 12x8 英寸
    plt.figure(figsize=(12, 8))
    # 绘制原始回报曲线，透明度 alpha=0.5
    plt.plot(episodes_list, return_list, label='raw', alpha=0.5)
    # 绘制滑动平均回报曲线
    plt.plot(episodes_list, mv_return_list, label='moving ave')
    # 设置 x 轴标签
    plt.xlabel('Episodes')
    # 设置 y 轴标签
    plt.ylabel('Returns')
    # 设置标题，显示代理名称和奖励类型
    # agent._get_name()：获取代理名称（例如 "REINFORCE"）
    # reward_type：未定义的变量，可能是遗留代码，需定义（如 "default"）
    reward_type = "default"  # 假设默认值，避免错误
    plt.title(f'{agent._get_name()} on RollingBall with {reward_type} reward')
    # 显示图例
    plt.legend()
    # 保存图像到 result 目录
    plt.savefig(f'./CartPole/result/{agent._get_name()}({reward_type}).png')
    # 显示图像
    plt.show()   