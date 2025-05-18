import gym
import gym

# 观测包装，把环境的原生二维观测转为一维的
class HashPosition(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env) # 调用 super().__init__(env) 完成包装器初始化
        self.env = env # 这里的env是原生的gym环境
        map_size = env.observation_space['agent'].nvec
        self.observation_space = gym.spaces.Discrete(map_size[0]*map_size[1]) # 新的观测空间
	
    def observation(self, obs):
        # 列优先（x 方向优先）展开的编号方式
        return obs["agent"][0] * self.env.nrow + obs["agent"][1]
