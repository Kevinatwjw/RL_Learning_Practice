import numpy as np
import gym
from collections import deque
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class ReplayBuffer:
    def __init__(self, max_batch_size):
        # deque定义的是一个先入先出对列，其效率高于.pop(0)
        self.max_batch_size = max_batch_size
        self.buffer = deque(maxlen=self.max_batch_size)  # 固定容量
        
    def push_transition(self, transition):
        # 这里的实现没有判断是否重复（去重操作适合探索性任务）
        self.buffer.append(transition)
        
    def sample(self, batch_size):
        if batch_size > self.max_batch_size:
            raise ValueError("采样的长度大于经验回放池大小！")
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def len_buffer(self):
        return len(self.buffer)
    
class Q_net(torch.nn.Module):
    """"Q网络使两个MLP进行连接，用于DQN和Double DQN"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        # 调用nn.Module的初始化，确保网络能够正确初始化
        super().__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        # 一维批归一化（LayerNorm Normalization）层的操作
        self.ln1 = nn.LayerNorm(hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)
        self._init_weights()
        
    def _init_weights(self):
        """对每一层进行权重初始化"""
        # 隐藏层：使用 ReLU 专用的 Kaiming 初始化
        nn.init.kaiming_uniform_(self.fc1.weight, nonlinearity='relu')
        # 初始化全连接层偏置，稳定初始输出，加速早期训练。
        nn.init.zeros_(self.fc1.bias)
        nn.init.kaiming_uniform_(self.fc2.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc2.bias)
        # 输出层受限初始化
        bound = 3 / np.sqrt(self.fc3.weight.size(0))
        nn.init.uniform_(self.fc3.weight, -bound, bound)
        nn.init.zeros_(self.fc3.bias)
        
    def forward(self, x):
        """定义从输入状态到输出 Q 值的计算流程。
            输入 x：状态张量，形状为 (batch_size, input_dim)
        """
        x = self.ln1(F.relu(self.fc1(x)))
        x = self.ln2(F.relu(self.fc2(x)))
        return self.fc3(x)

class DQN(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, action_range, lr, gamma, 
                 epsilon, device, seed=None):
        super().__init__()
        # 初始化参数
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_range = action_range # 动作范围
        self.epsilon = epsilon # spsilon-greedy探索率
        self.gamma = gamma # 折扣率
        self.lr = lr
        self.rng = np.random.RandomState(seed) 
        self.device = device
        # 初始化训练网络
        self.Qnet = Q_net(state_dim, hidden_dim, action_dim).to(device)
        # 初始化目标网络
        self.Qnet_target = Q_net(state_dim, hidden_dim, action_dim).to(device)
        # 初始化使用Adam更新器(仅优化在线训练网络)
        self.optimizer = torch.optim.Adam(self.Qnet.parameters(), lr=lr)
    
    def max_q_value_of_state(self, state):
        """计算给定状态下所有动作的最大的Q值，评估策略质量"""
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        return self.Qnet(state).max().item()

    def take_action(self, state):
        """采取动作,使用ε-greedy策略"""
        if self.rng.random() < self.epsilon:
            action = self.rng.randint(self.action_range)
        else:
            state = torch.tensor(state, dtype=torch.float32,).to(self.device)
            action = self.Qnet(state).argmax().item()
        return action
    
    def update(self, transition_dict, tau=0.01):
        # 将经验转化成张量：states,next_states-[batch_size, state_dim],actions-[batch_size, 1],rewards,done-[(batch_size)]
        states = torch.tensor(transition_dict['states'], dtype=torch.float32).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.int64).view(-1, 1).to(self.device)
        # 创建张量->升维，[batch_size, 1]（该形式符合自动广播机制）->设备转移->降维tensor([1,2,3])
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float32).view(-1,1).to(self.device).squeeze()
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float32).view(-1,1).to(self.device).squeeze()
        
        # Q值计算and目标Q值计算
        # 计算当前状态-动作对的 Q 值 Q(s_i, a_i; θ)。
        q_values = self.Qnet(states).gather(dim=1, index = actions).squeeze()
        max_next_q_valus = self.Qnet_target(next_states).max(axis=1)[0] # 取每行最大值
        # 结合未来奖励与未来Q值，并考虑是否终止
        q_targets = rewards + self.gamma * max_next_q_valus * (1 - dones)
        
        # 计算训练损失(q_values，q_targets均为（batch_size,actions_dim）)
        DQN_Loss = F.mse_loss(q_values, q_targets, reduction='mean') # 计算 Q 值与目标的 MSE.（默认自动平均）
        self.optimizer.zero_grad() # 清空梯度,避免梯度累加（PyTorch默认会累加梯度，训练时必须手动清零）
        DQN_Loss.backward() # 自动计算损失对网络参数的梯度,梯度存储在参数的 .grad 属性中
        self.optimizer.step() # 根据梯度更新网络
        
        # 目标网络更新(采用软更新)(网络中使用layernorm，如果使用batchnorm需要将running.mean和runnig.var进行软更新)
        for target_param, param in zip(self.Qnet_target.parameters(), self.Qnet.parameters()):
            target_param.data.copy_(tau * param + (1-tau)*target_param)
        """
        若网络同时包含batchnorm和layernorm:使用一下方案
        for target_param, param in zip(self.Qnet_target.modules(), self.Qnet.modules()):
            # 更新普通参数
            if hasattr(target_param, 'weight') and not isinstance(target_param, (nn.BatchNorm1d, nn.LayerNorm)):
                target_param.weight.data.copy_(tau*param.weight + (1-tau)*target_param.weight)
            # 更新BatchNorm的running统计量
            if isinstance(tgtarget_paramt, nn.BaatchNorm1d):
                target_param.running_mean.copy_(tau*param.running_mean + (1-tau)*target_param.running_mean)
                target_param.running_var.copy_(tau*param.running_var + (1-tau)*target_param.running_var)
        """
            