import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np

class PolicyNet(torch.nn.Module):
    """策略网络:需要输入state_dim,输出的是action_dim的概率，即每个动作选择的概率"""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, output_dim)

        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        # 加入dim=1，表示在第二个维度上进行softmax操作，即在每个样本上进行softmax操作
        x = F.relu(self.fc2(x))
        return F.softmax(self.fc3(x), dim=1) # 转化成概率输出(1, output_dim)
    
class ValueNet(torch.nn.Module):
    """输出的是当前状态s的状态价值，输出的维度为1"""
    def __init__(self, input_dim, hidden_dim):
        super(ValueNet, self).__init__()  
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = torch.nn.Linear(hidden_dim, 1)

    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
    

class REINFORCE_Baseline(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, gamma, lr_policy, lr_value, device):
        super(REINFORCE_Baseline, self).__init__()
        # 先定义网络，并将网络送到GPU
        self.policynet = PolicyNet(state_dim, hidden_dim, action_dim).to(device)
        self.valuenet = ValueNet(state_dim, hidden_dim).to(device)
        # 在定义对应网络的优化器
        self.optimizer_policy = torch.optim.Adam(self.policynet.parameters(), lr=lr_policy)
        self.optimizer_valuenet = torch.optim.Adam(self.valuenet.parameters(), lr=lr_value)
        self.gamma = gamma
        self.device = device
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # 在第一个维度上增加维度，将state形状变成（1, state_dim）,为了和网络输出的维度匹配
        state = state.unsqueeze(0)
        # 将state送到策略网络中并将维度变成形状为（1，action_dim）
        probs = self.policynet(state).squeeze()
        # 使用torch.distributions.Categorical()将probs转化成概率分布
        action_dist = torch.distributions.Categorical(probs)
        # 按照概率分布随机采样一个动作
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        G, returns = 0, []
        for reward in reversed(transition_dict['rewards']):
            # 公式：G_t = r_t + gamma * G_t+1
            G = self.gamma * G + reward
            # 注意一个现象，G_t随着t增加而减小，因为开始的越早，其未来收益就越多（因为G_t = r_t + gamma * G_t+1）
            returns.insert(0, G)
        # 将各个变量送进GPU
        # states原本是list，需要转换成numpy数组，因为numpy数组可以被torch.tensor()识别
        states = torch.tensor(np.array(transition_dict['states']), dtype=torch.float).to(self.device) 
        # 将actions转换成tensor，并增加一个维度，变成（batch_size,action_dim）
        actions = torch.tensor(transition_dict['actions']).view(-1,1).to(self.device)
        # 将rewards转换成tensor，并增加一个维度，变成（batch_size,1）,传送给GPU后再去掉多余维度
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device).squeeze()
        # returns转换成tensor，并增加一个维度，变成（batch_size,1）,传送给GPU后再去掉多余维度
        returns = torch.tensor(returns, dtype=torch.float).view(-1,1).to(self.device).squeeze()
        
        # 梯度清零
        self.optimizer_policy.zero_grad()
        self.optimizer_valuenet.zero_grad()
        
        # 更新价值网络 ,states是（batch_size,state_dim）,value_predicets是（batch_size,1）
        value_predicts = self.valuenet(states).squeeze()
        # 使用均方误差计算与真实值的误差，这是蒙特卡洛采样的体现
        value_loss = torch.mean(F.mse_loss(value_predicts, returns))
        # 反向传播
        value_loss.backward()
        # 更新价值网络
        self.optimizer_valuenet.step()
        
        # 更新策略网络，从轨迹最后一时刻开始，每步回传累计梯度（为了对齐，梯度累计无关顺序）
        for i in reversed(range(len(rewards))):
            state = states[i]
            action = actions[i]
            value = self.valuenet(state).squeeze()
            G = returns[i]
            # 得到该状态下的动作概率，形状是（action_dim,）
            probs = self.policynet(state.unsqueeze(0)).squeeze()
            # 将概率取成对数概率(已知所选择的动作，直接操作对应的动作)
            log_probs = torch.log(probs[action])
            # 计算策略的误差,使用detach()将value从计算出隔离，防止被valuenet的梯度被影响
            policy_loss = -log_probs * (G - value.detach())
            # 反向传播，累积梯度
            policy_loss.backward()
        # 一步更新总的策略网络参数
        self.optimizer_policy.step()
            
class A2C(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma, device):
        super().__init__()
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.device = device
        
        # 定义策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        # 定义优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # 扩展state的维度，变成（1,state_dim）
        state = state.unsqueeze(0)
        probs = self.actor(state).squeeze()
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        # 从transtions中提取states,actions,rewards,next_states,dones
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        
        # 计算网络的损失
        # Critic网络的损失,用 mse loss 去优化 Sarsa TD error
        # 公式：td_target = r_t + gamma * V(s_t+1)
        td_target = rewards + self.gamma*self.critic(next_states)*(1-dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        
        # Actor网络的损失，使用log_prob去优化TD error
        # 公式：log_prob(a_t|s_t) * (G_t - V(s_t))
        td_error = td_target - self.critic(states)
        # 使用gather()将actions转换成one-hot编码，并计算log_prob
        # one-hot编码：[0,1,0,0]
        # probs的形状是（batch_size, action_dim），actions的形状是（batch_size,1）
        probs = self.actor(states).gather(1, actions)
        log_probs = torch.log(probs)
        # 最小化
        actor_loss = torch.mean(-log_probs * td_error.detach())
        
        # 梯度清零
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # 反向传播
        actor_loss.backward()
        critic_loss.backward()
        
        # 更新网络
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        
class A2C_target(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, actor_lr, critic_lr, gamma,target_weight, device):
        super().__init__()
        self.gamma = gamma
        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.target_weight = target_weight
        self.device = device
        
        # 定义策略网络
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.critic_target = ValueNet(state_dim, hidden_dim).to(self.device)
        # 定义优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
        
    def take_action(self, state):
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        # 扩展state的维度，变成（1,state_dim）
        state = state.unsqueeze(0)
        probs = self.actor(state).squeeze()
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def update(self, transition_dict):
        # 从transtions中提取states,actions,rewards,next_states,dones
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        
        # 计算网络的损失
        # Critic网络的损失,用 mse loss 去优化 Sarsa TD error
        # 公式：td_target = r_t + gamma * V(s_t+1)
        td_target = rewards + self.gamma*self.critic_target(next_states)*(1-dones)
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        
        # Actor网络的损失，使用log_prob去优化TD error
        # 公式：log_prob(a_t|s_t) * (G_t - V(s_t))
        td_error = td_target - self.critic(states)
        # 使用gather()将actions转换成one-hot编码，并计算log_prob
        # one-hot编码：[0,1,0,0]
        # probs的形状是（batch_size, action_dim），actions的形状是（batch_size,1）
        probs = self.actor(states).gather(1, actions)
        log_probs = torch.log(probs)
        # 最小化
        actor_loss = torch.mean(-log_probs * td_error.detach())
        
        # 梯度清零
        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        
        # 反向传播
        actor_loss.backward()
        critic_loss.backward()
        
        # 更新网络
        self.actor_optimizer.step()
        self.critic_optimizer.step()
        
        # 更新目标网络
        params_target = list(self.critic_target.parameters())
        params_critic = list(self.critic.parameters())
        for i in range(len(params_target)):
            new_params_target = self.target_weight * params_target[i] + (1-self.target_weight) * params_critic[i]
            params_target[i].data.copy_(new_params_target)
            
            