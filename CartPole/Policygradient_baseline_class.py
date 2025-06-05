import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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
    

