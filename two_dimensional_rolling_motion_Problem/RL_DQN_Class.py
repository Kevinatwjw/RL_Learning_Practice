import numpy as np
import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    ''' 经验回放池 '''
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)        # 先进先出队列

    def add(self, state, action, reward, next_state, done):  
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):  
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done

    def size(self): 
        return len(self.buffer)

    
class Q_Net(torch.nn.Module):
    ''' Q 网络是一个两层 MLP, 用于 DQN 和 Double DQN '''
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """使用Kaiming初始化权重，适合激活函数为Relu"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用 Kaiming 均匀初始化，指定 nonlinearity='relu'
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                # 偏置置零
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
                    
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class DQN(torch.nn.Module):
    ''' DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, tau, device, seed=None):
        super().__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_range = action_range        # action 取值范围
        self.gamma = gamma                      # 折扣因子
        self.epsilon = epsilon                  # epsilon-greedy
        self.tau = tau
        self.rng = np.random.RandomState(seed)  # agent 使用的随机数生成器
        self.device = device                
        
        # Q 网络
        self.q_net = Q_Net(state_dim, hidden_dim, action_range).to(self.device)  
        # 目标网络
        self.target_q_net = Q_Net(state_dim, hidden_dim, action_range).to(self.device)
        # 使用Adam优化器
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        
    def max_q_value_of_given_state(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        return self.q_net(state).max().item()
        
    def take_action(self, state):  
        ''' 按照 epsilon-greedy 策略采样动作 '''
        if self.rng.random() < self.epsilon:
            action = self.rng.randint(self.action_range)
        else:
            state = torch.tensor(state, dtype=torch.float).to(self.device)
            action = self.q_net(state).argmax().item()
        return action

    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)                             # (bsz, state_dim)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)                   # (bsz, state_dim)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)               # (bsz, act_dim)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()     # (bsz, )
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()         # (bsz, )

        q_values = self.q_net(states).gather(dim=1, index=actions).squeeze()                # (bsz, )
        max_next_q_values = self.target_q_net(next_states).max(axis=1)[0]                   # (bsz, )
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)                  # (bsz, )

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.optimizer.zero_grad()                                                         
        dqn_loss.backward() 
        self.optimizer.step()
        
        # 软更新目标网络参数
        for target_param, q_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)


class DoubleDQN(DQN):
    ''' Double DQN算法 '''
    def __init__(self, state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, device, tau=0.001, seed=None):
        super().__init__(state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, device, tau, seed)
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)                             # (bsz, state_dim)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)                   # (bsz, state_dim)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)               # (bsz, act_dim)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()     # (bsz, )
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1, 1).to(self.device).squeeze()         # (bsz, )
        # Double DQN：主网络选择动作，目标网络估计Q值
        q_values = self.q_net(states).gather(dim=1, index=actions).squeeze()                # (bsz, )
        # 使用Q网络估计最优动作（[0]取最优值，[1]取最优值的索引）
        max_actions_index = self.q_net(next_states).max(axis=1)[1]
        # 由目标网络计算Q值
        max_next_q_values = self.target_q_net(next_states).gather(dim=1, index = max_actions_index.unsqueeze(1)).squeeze()                   # (bsz, )
        q_targets = rewards + self.gamma * max_next_q_values * (1 - dones)                  # (bsz, )

        dqn_loss = torch.mean(F.mse_loss(q_values, q_targets))  
        self.optimizer.zero_grad()                                                         
        dqn_loss.backward() 
        self.optimizer.step()
        
        # 软更新目标网络参数
        for target_param, q_param in zip(self.target_q_net.parameters(), self.q_net.parameters()):
            target_param.data.copy_(self.tau * q_param.data + (1.0 - self.tau) * target_param.data)
            
class VA_net(torch.nn.Module):
    """_summary_

    Args:
        torch (_type_): _description_
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(VA_net, self).__init__()
        # 共享网络部分
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2= torch.nn.Linear(hidden_dim, hidden_dim)
        # 输出每个动作的优势值A(s,a)，维度为动作空间的大小
        self.fc_A = torch.nn.Linear(hidden_dim, output_dim)
        # 输出状态价值V(s),维度为1
        self.fc_V = torch.nn.Linear(hidden_dim, 1)
        self._init_weights()
    
    def _init_weights(self):
        """使用Kaiming初始化权重，适合激活函数为Relu"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # 使用 Kaiming 均匀初始化，指定 nonlinearity='relu'
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                # 偏置置零
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        x = F.relu(self.fc2(x))
        A = self.fc_A(x)
        V = self.fc_V(x)
        Q = V + (A - A.mean().item())
        return Q
    
class DuelingDON(DQN):
    def __init__(self, state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, device, tau=0.001, seed=None):
        super().__init__(state_dim, hidden_dim, action_dim, action_range, lr, gamma, epsilon, device, tau, seed)
        
        self.q_net = VA_net(state_dim, hidden_dim, action_range).to(self.device)
        self.target_q_net = VA_net(state_dim, hidden_dim, action_range).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr = lr)
        
class PolicyNet(torch.nn.Module):
    """
    策略网络，用于强化学习中的策略梯度方法（如 REINFORCE）。
    网络结构为三层 MLP（多层感知机），输入状态，输出动作概率分布。
    """
    
    def __init__(self, input_dim, hidden_dim, output_dim):
        """
        初始化策略网络。

        参数：
        - input_dim (int): 输入维度，即状态空间的维度。
        - hidden_dim (int): 隐藏层维度，控制网络容量。
        - output_dim (int): 输出维度，即动作空间的大小（离散动作）。

        网络结构：
        - fc1: 输入层 -> 隐藏层 1
        - fc2: 隐藏层 1 -> 隐藏层 2
        - fc3: 隐藏层 2 -> 输出层
        """
        super(PolicyNet, self).__init__()  # 调用父类构造函数，初始化 torch.nn.Module
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)  # 第一层线性变换：输入层 -> 隐藏层 1
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)  # 第二层线性变换：隐藏层 1 -> 隐藏层 2
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)  # 第三层线性变换：隐藏层 2 -> 输出层（动作概率）
        self._init_weights()  # 调用权重初始化方法，设置初始参数

    def _init_weights(self):
        """
        自定义权重初始化方法，确保网络初始行为适合强化学习任务。

        目标：
        - 隐藏层：使用 Kaiming 初始化，适合 ReLU 激活函数，保持激活值和梯度方差稳定。
        - 输出层：使用小方差正态分布，确保初始动作概率分布接近均匀，促进探索。
        """
        # 隐藏层 1 (fc1) 权重初始化
        # 使用 Kaiming 初始化（正态分布形式），适合 ReLU 激活函数
        # mode='fan_in'：方差计算基于输入神经元数量，避免激活值方差过大或过小
        # nonlinearity='relu'：指定激活函数为 ReLU，调整方差为 2/fan_in
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        
        # 隐藏层 1 (fc1) 偏置初始化
        # 初始化为 0.1（小正值），确保 ReLU 激活后更多神经元活跃（输出非 0）
        # 避免“神经元死亡”（ReLU 输出恒为 0），提升网络初始表达能力
        torch.nn.init.constant_(self.fc1.bias, 0.1)
        
        # 隐藏层 2 (fc2) 权重初始化
        # 同 fc1，使用 Kaiming 初始化，保持深层网络中激活值和梯度的稳定性
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        # 隐藏层 2 (fc2) 偏置初始化
        # 同 fc1，设置为 0.1，增加神经元活跃性
        torch.nn.init.constant_(self.fc2.bias, 0.1)
        
        # 输出层 (fc3) 权重初始化
        # 使用小方差正态分布（均值 0，标准差 0.01），使初始权重接近 0
        # 这样，fc3 层的输出 z = Wx + b 接近 0，softmax(z) 接近均匀分布（1/output_dim）
        # 适合强化学习（如 REINFORCE）中初始探索需求，避免过早收敛到次优策略
        torch.nn.init.normal_(self.fc3.weight, mean=0, std=0.01)
        
        # 输出层 (fc3) 偏置初始化
        # 设置为 0，确保初始动作概率分布完全由权重决定，避免引入额外偏移
        torch.nn.init.zeros_(self.fc3.bias)

    def forward(self, x):
        """
        前向传播，计算状态对应的动作概率分布。

        参数：
        - x (torch.Tensor): 输入张量，形状为 (batch_size, input_dim)，表示一批状态。

        返回：
        - torch.Tensor: 动作概率分布，形状为 (batch_size, output_dim)，表示每个动作的概率。
        """
        # 第一层：线性变换 + ReLU 激活
        # 输入 x 的形状为 (batch_size, input_dim)
        # 输出形状为 (batch_size, hidden_dim)
        # ReLU(z) = max(0, z)，激活函数增加非线性，截断负值
        x = F.relu(self.fc1(x))
        
        # 第二层：线性变换 + ReLU 激活
        # 输入形状为 (batch_size, hidden_dim)
        # 输出形状为 (batch_size, hidden_dim)
        # 进一步增加网络深度和非线性表达能力
        x = F.relu(self.fc2(x))
        
        # 第三层：线性变换 + Softmax 激活
        # 输入形状为 (batch_size, hidden_dim)
        # 输出 z 的形状为 (batch_size, output_dim)，表示每个动作的未归一化得分（logits）
        # Softmax 将 logits 转换为概率分布，确保 sum(probs) = 1
        # 数学上：softmax(z)_i = exp(z_i) / sum(exp(z_j))
        # dim=1 表示在动作维度上归一化
        return F.softmax(self.fc3(x), dim=1)
    
class REINFORCE(torch.nn.Module):
    """
    REINFORCE 算法实现，基于策略梯度方法，用于强化学习任务。
    该类包含策略网络（PolicyNet）和优化器，通过蒙特卡洛方法估计梯度并更新策略参数。
    """
    
    def __init__(self, state_dim, hidden_dim, action_range, learning_rate, gamma, device):
        """
        初始化 REINFORCE 算法。

        参数：
        - state_dim (int): 状态空间的维度（输入维度）。
        - hidden_dim (int): 策略网络隐藏层的维度，控制网络容量。
        - action_range (int): 动作空间的大小（离散动作数量，即输出维度）。
        - learning_rate (float): 学习率，用于优化器（Adam）。
        - gamma (float): 折扣因子，用于计算折扣回报，范围 [0, 1]。
        - device (torch.device): 计算设备（CPU 或 GPU，如 torch.device('cuda')）。
        """
        super().__init__()  # 调用父类 torch.nn.Module 的构造函数，初始化模块
        # 初始化策略网络 PolicyNet，用于参数化策略 π_θ(a|s)
        # state_dim -> hidden_dim -> action_range
        self.policynet = PolicyNet(state_dim, hidden_dim, action_range).to(device)
        # 使用 Adam 优化器优化策略网络参数
        # lr=learning_rate 指定学习率 α，用于梯度上升更新参数
        self.optimizer = torch.optim.Adam(self.policynet.parameters(), lr=learning_rate)
        # 折扣因子 γ，用于计算折扣回报 G_t
        self.gamma = gamma
        # 计算设备，确保张量和网络在同一设备上（CPU 或 GPU）
        self.device = device
        
    def take_action(self, state):
        """
        根据当前策略 π_θ(a|s) 从给定状态中采样动作。

        参数：
        - state (list or np.ndarray): 当前状态，通常是一个一维数组，形状为 (state_dim,)。

        返回：
        - int: 采样得到的动作索引（离散动作）。
        """
        # 将输入状态转换为 PyTorch 张量，并指定数据类型为浮点数
        # 示例：state = [0.5, 0.3, 1.0, -0.2] -> tensor([0.5, 0.3, 1.0, -0.2])
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # 增加批次维度，形状从 (state_dim,) 变为 (1, state_dim)
        # 神经网络（如 PolicyNet）通常期望输入是批量形式 (batch_size, input_dim)
        # 示例：tensor([0.5, 0.3, 1.0, -0.2]) -> tensor([[0.5, 0.3, 1.0, -0.2]])
        state = state.unsqueeze(0)
        # 通过策略网络计算动作概率分布 π_θ(a|s)，输出形状为 (1, action_range)
        # squeeze() 移除批次维度，形状变为 (action_range,)
        # 示例：tensor([[0.33, 0.33, 0.34]]) -> tensor([0.33, 0.33, 0.34])
        probs = self.policynet(state).squeeze()
        # 使用分类分布（Categorical Distribution）表示离散动作概率分布
        # torch.distributions.Categorical 需要一维概率向量，probs 的和为 1（由 softmax 保证）
        action_dist = torch.distributions.Categorical(probs)
        # 从概率分布中采样一个动作，action 是一个标量张量
        # 示例：若 probs=[0.33, 0.33, 0.34]，action 可能是 tensor(2)（以 0.34 的概率采样到动作 2）
        action = action_dist.sample()
        # 将张量转换为 Python 整数，方便传递给环境
        # 示例：tensor(2) -> 2
        return action.item()
    
    def update(self, transition_dict):
        """
        根据一条轨迹更新策略网络参数，使用 REINFORCE 算法的策略梯度方法。

        参数：
        - transition_dict (dict): 包含一条轨迹的数据，键包括：
            - 'rewards': 奖励列表 [r_0, r_1, ..., r_T]
            - 'states': 状态列表 [s_0, s_1, ..., s_T]
            - 'actions': 动作列表 [a_0, a_1, ..., a_T]

        数学原理：
        - 目标函数 J(θ) = E[R(τ)]，其中 R(τ) 是轨迹总回报
        - 策略梯度定理：∇_θ J(θ) = E[Σ_t ∇_θ log π_θ(a_t|s_t) * G_t]
        - G_t 是从时间步 t 开始的折扣回报：G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
        """
        # 从字典中提取相关变量
        # 奖励列表 [r_0, r_1, ..., r_T]
        reward_list = transition_dict['rewards']
        # 状态列表 [s_0, s_1, ..., s_T]
        state_list = transition_dict['states']
        # 动作列表 [a_0, a_1, ..., a_T]
        action_list = transition_dict['actions']
        # 初始化折扣回报 G 为 0，用于递归计算 G_t
        G = 0
        # 清空优化器的梯度缓存，为本次更新准备
        # 避免上一次更新的梯度干扰
        self.optimizer.zero_grad()
        
        # 从轨迹最后一步（t=T）向前遍历到第一步（t=0）
        # 逆序计算折扣回报 G_t = r_t + γ r_{t+1} + γ^2 r_{t+2} + ...
        # reversed() 反转序列，例如 range(5) -> [4, 3, 2, 1, 0]
        for i in reversed(range(len(reward_list))):
            # 获取当前时间步 t=i 的奖励 r_t
            # 示例：reward_list=[1.0, 0.5, -0.2, 2.0, 3.0]，i=4 -> reward=3.0
            reward = reward_list[i]
            # 将当前状态 s_t 转换为 PyTorch 张量，并移动到指定设备
            # 形状为 (state_dim,)，例如 tensor([0.5, 0.3, 1.0, -0.2])
            state = torch.tensor(state_list[i], dtype=torch.float).to(self.device)
            # 增加批次维度，形状从 (state_dim,) 变为 (1, state_dim)
            # 适配 PolicyNet 的输入要求
            # 然后计算动作概率分布 π_θ(a|s_t)，输出形状为 (1, action_range)
            # squeeze() 移除批次维度，形状变为 (action_range,)
            # 示例：probs = tensor([0.33, 0.33, 0.34])
            probs = self.policynet(state.unsqueeze(0)).squeeze()
            # 获取当前时间步的动作 a_t
            # 示例：action_list=[0, 1, 2, 1, 0]，i=4 -> action=0
            action = action_list[i]
            # 计算动作 a_t 的对数概率 log π_θ(a_t|s_t)
            # probs[action] 获取第 action 个动作的概率
            # torch.log() 计算自然对数
            # 示例：若 probs[0]=0.33，则 log_prob=torch.log(0.33)≈-1.1086
            log_prob = torch.log(probs[action])
            # 递归计算折扣回报 G_t
            # G_t = r_t + γ G_{t+1}
            # 示例：若 gamma=0.99，G=0（初始），reward=3.0，则 G=3.0
            G = self.gamma * G + reward  
            # 计算当前时间步的损失
            # loss = -log π_θ(a_t|s_t) * G_t
            # 负号是因为 PyTorch 优化器执行梯度下降，而我们需要梯度上升
            # 梯度：∇_θ loss = -∇_θ log π_θ(a_t|s_t) * G_t，与策略梯度定理一致
            loss = -log_prob * G
            # 反向传播，计算损失对策略网络参数的梯度
            # 梯度会累积到 self.policynet.parameters() 的 .grad 属性中
            loss.backward()
        
        # 使用累积的梯度更新策略网络参数
        # Adam 优化器执行一步梯度下降：θ = θ - α * ∇_θ loss
        # 由于 loss 中有负号，实际执行的是梯度上升：θ = θ + α * ∇_θ J(θ)
        self.optimizer.step()
        
        
        
        
