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
        self.fc2 = torch.nn.Linear(hidden_dim, output_dim)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc1.bias)
        torch.nn.init.xavier_normal_(self.fc2.weight)
        torch.nn.init.zeros_(self.fc2.bias)
        
    def forward(self, x):
        x = F.relu(self.fc1(x)) 
        # 加入dim=1，表示在第二个维度上进行softmax操作，即在每个样本上进行softmax操作
        return F.softmax(self.fc2(x), dim=1) # 转化成概率输出(1, output_dim)
    
class ValueNet(torch.nn.Module):
    """输出的是当前状态s的状态价值，输出的维度为1"""
    def __init__(self, input_dim, hidden_dim):
        super(ValueNet, self).__init__()  
        self.fc1 = torch.nn.Linear(input_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
        torch.nn.init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc1.bias)
        # 在此处因为上一层输出激活为relu，如果使用xavier初始化，可能会带来正偏差】
        torch.nn.init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        torch.nn.init.zeros_(self.fc2.bias)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        return self.fc2(x)
    

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
            
        
        