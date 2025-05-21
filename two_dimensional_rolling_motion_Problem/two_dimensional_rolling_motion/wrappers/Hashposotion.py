import numpy as np
from gym import spaces
import gym

class DiscreteActionWrapper(gym.ActionWrapper):
    ''' 将 RollingBall 环境的二维连续动作空间离散化为二维离散动作空间 '''
    def __init__(self, env, bins):
        super().__init__(env)
        bin_width = 2.0 / bins
        self.action_space = spaces.MultiDiscrete([bins, bins]) 
        self.action_mapping = {i : -1+(i+0.5)*bin_width for i in range(bins)}

    def action(self, action):
        # 用向量化函数实现高效 action 映射
        vectorized_func = np.vectorize(lambda x: self.action_mapping[x])    
        result = vectorized_func(action)
        action = np.array(result)
        return action

class FlattenActionSpaceWrapper(gym.ActionWrapper):
    ''' 将多维离散动作空间拉平成一维动作空间 '''
    def __init__(self, env):
        super(FlattenActionSpaceWrapper, self).__init__(env)
        new_size = 1
        for dim in self.env.action_space.nvec:
            new_size *= dim
        self.action_space = spaces.Discrete(new_size)
    
    def action(self, action):
        orig_action = []
        for dim in reversed(self.env.action_space.nvec):
            orig_action.append(action % dim)
            action //= dim
        orig_action.reverse()
        return np.array(orig_action)
