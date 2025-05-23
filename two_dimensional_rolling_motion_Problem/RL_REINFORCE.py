import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
from gym.utils.env_checker import check_env
from RL_DQN_Class import ReplayBuffer, REINFORCE
from gym.wrappers import TimeLimit
from two_dimensional_rolling_motion import RollingBall, DiscreteActionWrapper, FlattenActionSpaceWrapper

if __name__ == "__main__":
    def moving_average(a, window_size):
        """
        生成序列 a 的滑动平均序列，用于平滑训练过程中的回报曲线。

        参数：
        - a (list or np.ndarray): 输入序列，通常是每回合的回报列表。
        - window_size (int): 滑动窗口大小，用于计算平均值。

        返回：
        - np.ndarray: 平滑后的序列，长度与 a 相同。

        实现：
        - 使用累积和（cumsum）计算滑动平均，处理首尾边界以保持序列长度。
        - 首尾部分使用逐渐增大的窗口，中间部分使用固定窗口。
        """
        # 计算累积和，插入 0 作为起始值，便于计算差分
        # 示例：a=[1,2,3,4,5] -> cumulative_sum=[0,1,3,6,10,15]
        cumulative_sum = np.cumsum(np.insert(a, 0, 0))
        # 中间部分：滑动平均 = (cumsum[i+window_size] - cumsum[i]) / window_size
        # 示例：window_size=3，middle=[(3-0)/3, (6-1)/3, (10-3)/3] -> [1, 1.67, 2.33]
        middle = (cumulative_sum[window_size:] - cumulative_sum[:-window_size]) / window_size
        # 首部：逐渐增大的窗口，例如 window_size=3，计算 a[0], a[0:2]
        r = np.arange(1, window_size-1, 2)  # [1, 3] for window_size=3
        begin = np.cumsum(a[:window_size-1])[::2] / r  # 例如 [a[0]/1, (a[0]+a[1])/3]
        # 尾部：逐渐减小的窗口，例如 window_size=3，计算 a[-1], a[-2:]
        end = (np.cumsum(a[:-window_size:-1])[::2] / r)[::-1]  # 例如 [(a[-2]+a[-1])/3, a[-1]/1]
        # 拼接首部、中间和尾部，得到平滑序列
        return np.concatenate((begin, middle, end))

    def set_seed(env, seed=42):
        """
        设置随机种子，确保实验可重复。

        参数：
        - env (gym.Env): 环境对象。
        - seed (int): 随机种子，默认为 42。

        作用：
        - 为环境、NumPy、Python 的 random 模块和 PyTorch 设置相同的随机种子。
        """
        # 设置环境动作空间的随机种子
        env.action_space.seed(seed)
        # 重置环境并设置种子
        env.reset(seed=seed)
        # 设置 Python 的 random 模块种子
        random.seed(seed)
        # 设置 NumPy 的随机种子
        np.random.seed(seed)
        # 设置 PyTorch 的随机种子，确保网络初始化和采样的一致性
        torch.manual_seed(seed)

    # 超参数定义
    state_dim = 4  # 环境状态维度，例如 RollingBall 环境的状态为 [x, y, vx, vy]
    action_dim = 1  # 环境动作维度（未离散化前），RollingBall 动作是一个二维向量 [ax, ay]
    action_bins = 10  # 动作离散化的 bins 数量，每个维度离散化为 10 个值
    action_range = action_bins * action_bins  # 离散化后的动作空间大小，例如 10*10=100
    learning_rate = 1e-3  # 学习率，用于 Adam 优化器，控制参数更新步长
    epsilon_start = 0.2  # 初始探索率（未使用，但可能是早期设计遗留）
    epsilon_end = 0.01  # 最终探索率（未使用，但可能是早期设计遗留）
    num_episodes = 1000  # 总训练回合数
    hidden_dim = 32  # 策略网络隐藏层维度，控制网络容量
    gamma = 0.98  # 折扣因子，用于计算折扣回报 G_t
    # 选择计算设备，若有 GPU 则使用 cuda，否则使用 CPU
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # 构建环境
    # RollingBall 环境：一个二维平面上的滚球任务，状态为 [x, y, vx, vy]，动作为 [ax, ay]
    env = RollingBall(render_mode='human', width=5, height=5, show_epi=True)
    # DiscreteActionWrapper：将连续动作空间离散化为二维离散动作空间，例如 [ax, ay] 每个维度离散化为 action_bins 个值
    # FlattenActionSpaceWrapper：将二维离散动作空间展平为一维动作空间，例如 (action_bins, action_bins) -> action_bins * action_bins
    env = FlattenActionSpaceWrapper(DiscreteActionWrapper(env, action_bins))
    # TimeLimit：限制每个回合的最大步数为 100，避免无限循环
    env = TimeLimit(env, 100)
    # 检查环境是否符合 gym 规范，例如动作和状态空间定义是否正确
    check_env(env.unwrapped)
    # 设置随机种子，确保实验可重复
    set_seed(env, 42)

    # 构建 REINFORCE 代理
    # state_dim=4：输入状态维度
    # hidden_dim=64：隐藏层维度
    # action_range=100：输出动作维度（离散动作数量）
    # learning_rate=1e-4：学习率
    # gamma=0.98：折扣因子
    # device：计算设备
    agent = REINFORCE(state_dim, hidden_dim, action_range, learning_rate, gamma, device)

    # 开始训练
    return_list = []  # 存储每个回合的总回报，用于后续绘图
    # 外循环：共 10 次迭代，每次迭代包含 num_episodes/10 回合
    for i in range(20):
        # 使用 tqdm 创建进度条，显示当前迭代进度
        # total=int(num_episodes/10)：每个迭代的回合数，例如 1000/20=50
        # desc='Iteration %d' % i：进度条描述，例如 "Iteration 0"
        with tqdm(total=int(num_episodes / 20), desc='Iteration %d' % i) as pbar:
            # 内循环：每个迭代包含 num_episodes/10 回合
            for i_episode in range(int(num_episodes / 10)):
                # 初始化回合总回报
                episode_return = 0
                # 存储轨迹数据的字典，包含状态、动作、下一状态、奖励和终止标志
                transition_dict = {
                    'states': [],
                    'actions': [],
                    'next_states': [],
                    'rewards': [],
                    'dones': []
                }
                # 重置环境，获取初始状态
                # state 是一个形状为 (state_dim,) 的数组，例如 [x, y, vx, vy]
                state, _ = env.reset()
                
                # 以当前策略交互，生成一条轨迹
                while True:
                    # 使用 REINFORCE 代理采样动作
                    # state 作为输入，输出一个离散动作索引（0 到 action_range-1）
                    action = agent.take_action(state)
                    # 执行动作，获取下一状态、奖励和终止标志
                    # next_state：下一状态，形状为 (state_dim,)
                    # reward：当前步的奖励，例如 -2.0（每步惩罚）
                    # terminated：是否到达目标（True 表示成功）
                    # truncated：是否超过最大步数（True 表示超时）
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    # 存储轨迹数据
                    transition_dict['states'].append(state)
                    transition_dict['actions'].append(action)
                    transition_dict['next_states'].append(next_state)
                    transition_dict['rewards'].append(reward)
                    transition_dict['dones'].append(terminated or truncated)
                    # 更新状态为下一状态
                    state = next_state
                    # 累加回合总回报
                    episode_return += reward

                    # 如果回合结束（成功或超时），渲染环境并退出循环
                    if terminated or truncated:
                        env.render()  # 渲染环境，显示轨迹（show_epi=True）
                        # 如果成功到达目标，打印信息
                        if terminated:
                            print(f"Episode {num_episodes / 20 * i + i_episode + 1}: Goal reached!")
                        break
                    # env.render()  # 注释掉的每步渲染，避免训练时过于频繁的显示
                    
                # 使用当前策略收集的轨迹数据进行 on-policy 更新
                # REINFORCE 是一种 on-policy 算法，更新时使用当前策略生成的数据
                # transition_dict 包含一条完整轨迹，agent.update 计算梯度并更新参数
                agent.update(transition_dict)
                # 记录当前回合的总回报
                return_list.append(episode_return)
                # 更新进度条，显示当前回合信息
                # episode：当前回合编号
                # return：当前回合总回报
                # ave return：最近 10 个回合的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return':
                    '%.3f' % episode_return,
                    'ave return':
                    '%.3f' % np.mean(return_list[-10:])
                })
                # 更新进度条
                pbar.update(1)
    # 训练结束后关闭环境
    env.close()
    
    # 创建结果目录，用于保存训练结果图
    # exist_ok=True：如果目录已存在，不会报错
    os.makedirs('./result', exist_ok=True)
    
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
    plt.savefig(f'./result/{agent._get_name()}({reward_type}).png')
    # 显示图像
    plt.show()