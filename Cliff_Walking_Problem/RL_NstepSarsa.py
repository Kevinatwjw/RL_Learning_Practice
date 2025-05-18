from CliffWalking import CliffWalkingEnv,HashPosition
from RL_Solver_Class import Nstep_SARSA
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# 用来设置一个 episode 的最⼤步数限制（step 数）。超过限制就自动终止 episode，返回 truncated = True。
from gym.wrappers import TimeLimit

# env实例化
env = CliffWalkingEnv(render_mode='human', map_size=(4, 12), pix_square_size=30)
env.action_space.seed(50)  # 设置随机种子
state, info = env.reset(seed=50)  # 重置环境，返回初始状态和信息
wrapper_env = TimeLimit(env, max_episode_steps=200)  # 设置最大步数限制
wrapper_env = HashPosition(wrapper_env)  # 使用哈希位置包装器将二维状态转化为一维状态


# Nstep_SARSA实例化
epsilon = 0.1           # 初始探索率
alpha = 0.1             # 初始学习率
agent = Nstep_SARSA(wrapper_env, alpha=alpha, gamma=0.9, epsilon=epsilon, seed=50, nstep=5)  # 创建Nstep_SARSA智能体

num_episodes = 1000  # 设置训练的总回合数
num_period = 100  # 设置轮数
#  Nstep_SARSA 算法本身内部有折扣因子 γ,γ 的影响体现在 Q 表的更新中；
return_list = []  # 用于存储每个周期的平均回报,回报G_t(G_t = r_t +r_{t+1} + ... +r_T)

 # 分轮完成训练，每轮结束后统计该轮平均回报 
for i in range(num_period):
    # # tqdm的进度条功能
    with tqdm(total = num_episodes / num_period, desc='Iteration %d' % i) as pbar:
        for episode in range(int(num_episodes / num_period)):
            # 重置环境
            episode_return = 0
            state, _ = wrapper_env.reset() # _表示不关心info的值
            action  = agent.take_action(state)  # 选择动作
            wrapper_env.render(
                # → (ncol, nrow)	转为二维矩阵（先按列）,-1表示自动计算
                state_values = agent.V_table.reshape(-1,wrapper_env.nrow).T, 
                policy = agent.greedy_policy  # 渲染策略
                )  # 渲染环境
            while True:
                next_state, reward, terminated, truncated, info = wrapper_env.step(action)  # 执行动作
                # 掉下悬崖或者到达终点，正常走
                next_action = 0 if terminated or truncated else agent.take_action(next_state)  # 选择下一个动作
                # 更新Q_table
                agent.update_Q_table(state, action, reward, next_state, next_action, done=(terminated or truncated))
                # 更新策略，贪婪策略
                agent.update_policy()
                # 更新状态价值
                agent.update_V_table()
                # 更新回报（注意非折扣）
                episode_return += reward
                # 若掉下悬崖或者到达终点，结束当前回合
                if terminated or truncated:
                    episode_return += reward
                    break
                # 更新状态和动作
                state = next_state
                action = next_action
            # 降低渲染频率，提高运算速度
            if episode % 5 == 0:
                wrapper_env.render(
                    # → (ncol, nrow)	转为二维矩阵（先按列）,-1表示自动计算
                    state_values = agent.V_table.reshape(-1,wrapper_env.nrow).T, 
                    policy = agent.greedy_policy  # 渲染策略
                    )
            return_list.append(episode_return)  # 记录回报
            if (episode + 1) % 5 == 0:  # 每10条序列打印一下这10条序列的平均回报
                pbar.set_postfix({
                    'episode':
                    '%d' % (num_episodes / 10 * i + episode + 1),
                    'return':
                    '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)
    agent.epsilon = max(0.01, agent.epsilon * 0.99)
    agent.alpha = max(0.01, agent.alpha * 0.99)
wrapper_env.close()

# 绘制return变化图
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.show()