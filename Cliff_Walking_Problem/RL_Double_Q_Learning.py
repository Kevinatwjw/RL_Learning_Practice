from CliffWalking import CliffWalkingEnv,HashPosition
from RL_Solver_Class import Double_Q_Learning
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# ��������һ�� episode ����?�������ƣ�step �������������ƾ��Զ���ֹ episode������ truncated = True��
from gym.wrappers import TimeLimit

# envʵ����
env = CliffWalkingEnv(render_mode='human', map_size=(4, 12), pix_square_size=30)
env.action_space.seed(42)  # �����������
state, info = env.reset(seed=42)  # ���û��������س�ʼ״̬����Ϣ
wrapper_env = TimeLimit(env, max_episode_steps=200)  # �������������
wrapper_env = HashPosition(wrapper_env)  # ʹ�ù�ϣλ�ð�װ������ά״̬ת��Ϊһά״̬


# Double Q-Learningʵ����
epsilon = 0.2           # ��ʼ̽����
alpha = 0.1             # ��ʼѧϰ��
agent = Double_Q_Learning(wrapper_env, alpha=alpha, gamma=0.9, epsilon=epsilon, seed=42)  # ����Double Q-Learning������

num_episodes = 1000  # ����ѵ�����ܻغ���
num_period = 100  # ��������
#  Double Q-Learning �㷨�����ڲ����ۿ����� ��,�� ��Ӱ�������� Q ��ĸ����У�
return_list = []  # ���ڴ洢ÿ�����ڵ�ƽ���ر�,�ر�G_t(G_t = r_t +r_{t+1} + ... +r_T)

 # �������ѵ����ÿ�ֽ�����ͳ�Ƹ���ƽ���ر� 
for i in range(num_period):
    # # tqdm�Ľ���������
    with tqdm(total = num_episodes / num_period, desc='Iteration %d' % i) as pbar:
        for episode in range(int(num_episodes / num_period)):
            # ���û���
            episode_return = 0
            state, _ = wrapper_env.reset() # _��ʾ������info��ֵ
            action  = agent.take_action(state)  # ѡ����
            wrapper_env.render(
                # �� (ncol, nrow)	תΪ��ά�����Ȱ��У�,-1��ʾ�Զ�����
                state_values = agent.V_table.reshape(-1,wrapper_env.nrow).T, 
                policy = agent.greedy_policy  # ��Ⱦ����
                )  # ��Ⱦ����
            while True:
                next_state, reward, terminated, truncated, info = wrapper_env.step(action)  # ִ�ж���
                # �������»��ߵ����յ㣬������
                next_action = 0 if terminated or truncated else agent.take_action(next_state)  # ѡ����һ������
                # ����Q_table
                agent.update_Q_table(state, action, reward, next_state, 0)
                # ���²��ԣ�̰������
                agent.update_policy()
                # ����״̬��ֵ
                agent.update_V_table()
                # ���»ر���ע����ۿۣ�
                episode_return += reward
                # ���������»��ߵ����յ㣬������ǰ�غ�
                if terminated or truncated:
                    break
                # ����״̬�Ͷ���
                state = next_state
                action = next_action
            # ������ȾƵ�ʣ���������ٶ�
            if episode % 5 == 0:
                wrapper_env.render(
                    # �� (ncol, nrow)	תΪ��ά�����Ȱ��У�,-1��ʾ�Զ�����
                    state_values = agent.V_table.reshape(-1,wrapper_env.nrow).T, 
                    policy = agent.greedy_policy  # ��Ⱦ����
                    )
            return_list.append(episode_return)  # ��¼�ر�
            if (episode + 1) % 5 == 0:  # ÿ10�����д�ӡһ����10�����е�ƽ���ر�
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

# ����return�仯ͼ
episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title(' Double Q-Learning on {}'.format('Cliff Walking'))
plt.show()