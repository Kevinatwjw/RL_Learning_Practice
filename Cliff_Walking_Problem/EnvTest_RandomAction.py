from CliffWalking import CliffWalkingEnv
from gym.utils.env_checker import check_env
import numpy as np
import random

map_size = (4,12)
env = CliffWalkingEnv(render_mode='human', 
					map_size=map_size, 
					pix_square_size=30)	# render_mode 设置为 'human' 以渲染价值颜色和贪心策略
print(check_env(env.unwrapped))         # 检查 base 环境是否符合 gym 规范
env.action_space.seed(42)				# 设置所有随机种子
observation, info = env.reset(seed=42)

for _ in range(10000):
	# 随机采样 action 执行一个 timestep
    observation, reward, terminated, truncated, info = env.step(env.action_space.sample())    
    
    # 随机产生状态价值和策略进行渲染
    env.render(state_values=np.random.randint(0, 10, map_size), 
                policy=np.array([np.array(random.sample(list(range(5)), random.randint(1, 5))) for _ in range(map_size[0]*map_size[1])], dtype=object))
    
    # 任务完成或失败，重置环境
    if terminated or truncated:
        print(reward, info)
        observation, info = env.reset()

env.close()
