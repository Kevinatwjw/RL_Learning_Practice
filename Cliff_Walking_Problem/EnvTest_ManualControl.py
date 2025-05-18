from CliffWalking import CliffWalkingEnv
from gym.utils.play import play
from gym.utils.env_checker import check_env
import pygame

map_size = (4,12)
env = CliffWalkingEnv(render_mode='rgb_array', 
					map_size=map_size, 
					pix_square_size=30) 	# 手动交互时渲染模式必须设置为 rgb_array 
print(check_env(env.unwrapped)) 			# 检查 base 环境是否符合 gym 规范
env.action_space.seed(42)					# 设置所有随机种子
observation, info = env.reset(seed=42)

# env.step() 后，env.render() 前的回调函数，可用来处理刚刚 timestep 中的运行信息
def playCallback(obs_t, obs_tp1, action, rew, terminated, truncated, info): 
    # 非 noop 动作，打印 reward 和附加 info
    if action != 4:
        print(f"reward: {rew}, info: {info}")

# key-action 映射关系 
mapping = {(pygame.K_UP,): 0, 
            (pygame.K_DOWN,): 1, 
            (pygame.K_LEFT,): 2, 
            (pygame.K_RIGHT,): 3}
        
# 开始手动交互
play(env, keys_to_action=mapping, callback=playCallback, fps=30, noop=4)

env.close()
