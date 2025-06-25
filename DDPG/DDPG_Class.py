import random
import gym
import numpy as np
from tqdm import tqdm
from collections import deque
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

class ReplayBuffer:
    """经验回放池"""
    def  __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) # 先进先出队列
    
    def add(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, done = zip(*transitions)
        return np.array(state), np.array(action), reward, np.array(next_state), done
    
    def size(self):
        return len(self.buffer)
    
    
class Policynet(torch.nn.Module):
    """
    DDPG的策略网络，输入状态，输出动作
    使用两层全连接网络，输出层使用tanh激活函数，输出动作
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound):
        super(Policynet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
        self.action_bound = action_bound
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        return torch.tanh(self.fc2(x)) * self.action_bound

class Qvaluenet(torch.nn.Module):
    """
    DDPG的Q值网络，输入状态和动作，输出Q值
    使用两层全连接网络，输出层使用线性激活函数，输出Q值
    """
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(Qvaluenet, self).__init__()
        # 输入动作和状态，这里区别于离散型，因为离散型动作都是已知，输出的是在一个状态下的Q值，而连续型动作是未知的，输出的是在状态和动作下的Q值
        self.fc1 = torch.nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        # dim=1表示在列的方向上进行拼接,在列的方向上是因为state和action都是列向量
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    
class DDPG(torch.nn.Module):
    """
    DDPG算法类
    """
    def __init__(self, state_dim, hidden_dim, action_dim, action_bound, gamma, tau, sigma, actor_lr, critic_lr, device):
        super(DDPG, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.gamma = gamma
        self.tau = tau
        self.sigma = sigma # 高斯噪声的标准差
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        
        self.actor = Policynet(state_dim, hidden_dim, action_dim, action_bound).to(self.device)
        self.critic = Qvaluenet(state_dim, hidden_dim, action_dim).to(self.device)
        self.target_actor = Policynet(state_dim, hidden_dim, action_dim, action_bound).to(self.device)
        self.target_critic = Qvaluenet(state_dim, hidden_dim, action_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())
        
    def soft_update(self, target, source):
        """软更新参数，将target网络的参数更新为source网络的参数"""
        for target_param, source_param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.tau) + source_param.data * self.tau)
            
    def take_action(self, state):
        """根据状态选择动作"""
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # .item()将tensor转换为标量
        action = self.actor(state).item()
        # 添加高斯噪声，使用np.random.randn(self.action_dim)生成一个与动作维度相同的随机数,范围在[-1,1]之间
        action = action + self.sigma * np.random.randn(self.action_dim)
        return action
    
    def update(self, transition_dict):
        """更新网络参数"""
        states = torch.tensor(transition_dict['states'],dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'],dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'],dtype=torch.float).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'],dtype=torch.float).view(-1,1).to(self.device)
        dones = torch.tensor(transition_dict['dones'],dtype=torch.float).view(-1,1).to(self.device)
       
        # 计算目标Q值，critic网络
        next_q_values = self.target_critic(next_states, self.target_actor(next_states))
        # .detach() 来切断梯度流，目标Q值被视为一个固定的数值目标
        q_targets = rewards + self.gamma * next_q_values.detach() * (1 - dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states, actions), q_targets))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 计算策略最大值，梯度上升，使用actor网络
        # 公式：J(theta) = -Q(s, pi(s))
        actor_loss = -torch.mean(self.critic(states, self.actor(states)))
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        self.soft_update(self.target_actor, self.actor)
        self.soft_update(self.target_critic, self.critic)
        
        