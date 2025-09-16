import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import copy

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    
def compute_advantage(gamma, lmbda, td_error):
    """计算优势函数（TRPO）,公式：A_t = 求和：gamma * lmbda * delta_t

    Args:
        gamma (_type_): 折扣系数
        lmbda (_type_): 优势函数的衰减系数
        td_error (_type_): 时序差分误差

    Returns:
        _type_: 优势函数
    """
    # .numpy()将tensor数组转化为numpy数组
    td_error = td_error.detach().numpy()
    advantage_list = []
    advantage = 0.0
    # 从前往后遍历td_error，其中td_error的形状是(batch_size,)，[::-1]表示从后往前遍历
    for delta in td_error[::-1]:
        # 计算优势函数，公式：A_t = gamma * lmbda * A_t+1 + delta_t
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)   

class PPO:
    """截断式PPO算法"""
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, lmbda, epochs, eps, gamma, device):
        self.gamma = gamma
        self.lmbda = lmbda
        self.epochs = epochs
        self.eps = eps
        self.device = device
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=critic_lr)
        
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        # (1)TD目标
        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        # (2)TD误差
        td_error = td_target - self.critic(states)
        # (3)估计优势函数
        advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)
        # (4)计算旧的策略对数分布
        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()
        
        # (5)对每个epoch进行计算和迭代
        for _ in range(self.epochs):
            # 计算新的策略对数分布(该过程中不断更新参数)
            log_probs = torch.log((self.actor(states)).gather(1,actions))
            # 计算概率比率
            ratio = torch.exp(log_probs - old_log_probs)
            # 计算截断的目标函数
            srr1 = ratio * advantage
            # clamp(x,l,r)=max(min(x,r),l)
            srr2 = torch.clamp(ratio, 1-self.eps, 1+self.eps) * advantage
            # 计算损失函数
            actor_loss = -torch.mean(torch.min(srr1, srr2))
            critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
            # 清空梯度累积
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            # 反向传播
            actor_loss.backward()
            critic_loss.backward()
            # 更新参数
            self.actor_optimizer.step()
            self.critic_optimizer.step()
            