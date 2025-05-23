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
