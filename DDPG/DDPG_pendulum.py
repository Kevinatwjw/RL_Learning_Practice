import numpy as np
import random
import torch
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from gym.utils.env_checker import check_env
from DDPG_Class import DDPG, ReplayBuffer
from gym.wrappers import TimeLimit

def moving_average(a, window_size):
    ''' 生成序列 a 的滑动平均序列 '''
    cumulative_sum = np.cumsum(np.insert(a, 0, 0))
    middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
    r = np.arange(1, window_size - 1, 2)
    begin = np.cumsum(a[:window_size - 1])[::2] / r
    end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]
    return np.concatenate((begin, middle, end))

def set_seed(env, seed=0):
    ''' 设置随机种子 '''
    env.action_space.seed(seed)
    env.reset(seed=seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

if __name__ == "__main__":
    # 环境配置
    env_name = 'Pendulum-v1'
    env = gym.make(env_name, render_mode='rgb_array')
    # env = TimeLimit(env, 1000)
    check_env(env.unwrapped)
    set_seed(env, 0)

    # 超参数
    state_dim = env.observation_space.shape[0]  # 3
    action_dim = env.action_space.shape[0]     # 1
    action_bound = env.action_space.high[0]    # 2.0
    actor_lr = 3e-2
    critic_lr = 3e-2
    num_episodes = 200
    hidden_dim = 128
    gamma = 0.98
    tau = 0.005
    buffer_size = 10000
    minimal_size = 5000
    batch_size = 128
    sigma = 0.1  # 初始噪声标准差
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 初始化
    replay_buffer = ReplayBuffer(buffer_size)
    agent = DDPG(state_dim, hidden_dim, action_dim, action_bound, sigma, actor_lr, critic_lr, tau, gamma, device)

    # 预填充回放缓冲区,原理是先随机探索，直到回放缓冲区达到最小容量
    state, _ = env.reset()
    while replay_buffer.size() <= minimal_size:
        action = np.clip(agent.take_action(state), -action_bound, action_bound)
        next_state, reward, terminated, truncated, _ = env.step(action)
        replay_buffer.add(state, action, reward, next_state, terminated or truncated)
        state = next_state
        if terminated or truncated:
            state, _ = env.reset()

    # 训练过程
    return_list = []
    for i in range(10):
        with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
            for i_episode in range(int(num_episodes / 10)):
                episode_return = 0
                state, _ = env.reset()

                # 收集轨迹并存储到回放缓冲区
                while True:
                    action = np.clip(agent.take_action(state), -action_bound, action_bound)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    replay_buffer.add(state, action, reward, next_state, done=terminated or truncated)

                    # 训练 Q 网络
                    assert replay_buffer.size() > minimal_size
                    # 确保缓冲区足够大
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(batch_size)
                    # 采样 128 条经验，组织批量数据
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
                    state = next_state
                    episode_return += reward
                    # 衰减噪声
                    agent.sigma = max(0.01, agent.sigma * 0.995)  # 指数衰减至 0.01
                    if terminated or truncated:
                        if terminated:
                            print(f"Episode {num_episodes / 10 * i + i_episode + 1}: Goal reached!")
                        break

 

                # 记录和更新进度
                return_list.append(episode_return)
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % episode_return,
                    'ave return': '%.3f' % np.mean(return_list[-10:])
                })
                pbar.update(1)

    # 关闭环境
    env.close()

    # 创建结果目录
    os.makedirs('./DDPG/result', exist_ok=True)

    # 可视化
    mv_return_list = moving_average(return_list, 29)
    episodes_list = list(range(len(return_list)))
    plt.figure(figsize=(12, 8))
    plt.plot(episodes_list, return_list, label='raw', alpha=0.5)
    plt.plot(episodes_list, mv_return_list, label='moving ave')
    plt.xlabel('Episodes')
    plt.ylabel('Returns')
    plt.title(f'{agent.__class__.__name__} on {env_name}')
    plt.legend()
    plt.savefig(f'./DDPG/result/{agent.__class__.__name__}.png')
    plt.show()