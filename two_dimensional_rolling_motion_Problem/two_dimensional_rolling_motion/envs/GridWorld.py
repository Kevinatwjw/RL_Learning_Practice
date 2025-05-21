import gym
from gym import spaces
import numpy as np
import pygame
import time

class RollingBall(gym.Env):
    # 环境支持的渲染模式和帧率设置
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 500,
    }

    def __init__(self, render_mode="human", width=10, height=10, show_epi=False):
        # 小球最大速度
        self.max_speed = 5.0
        self.width = width    # 环境宽度
        self.height = height  # 环境高度
        self.show_epi = show_epi  # 是否显示轨迹线

        # 定义动作空间：二维连续动作（作用力方向）
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float64)

        # 定义观测空间：[x位置, y位置, x速度, y速度]
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -self.max_speed, -self.max_speed]),
            high=np.array([width, height, self.max_speed, self.max_speed]),
            dtype=np.float64
        )

        # 小球初始速度和物理参数
        self.velocity = np.zeros(2, dtype=np.float64)
        self.mass = 0.005
        self.time_step = 0.01  # 时间步长

        # 奖励设计
        self.rewards = {
            'step': -2.0,     # 每一步惩罚
            'bounce': -10.0,  # 碰壁惩罚
            'goal': 300.0     # 达成目标奖励
        }

        # 设置目标和起点位置
        self.target_position = np.array([self.width * 0.8, self.height * 0.8], dtype=np.float32)
        self.start_position = np.array([width * 0.2, height * 0.2], dtype=np.float64)
        self.position = self.start_position.copy()

        # 渲染相关参数
        self.render_width = 300
        self.render_height = 300
        self.scale = self.render_width / self.width
        self.window = None
        self.clock = None

        # 存储轨迹
        self.trajectory = []

        # 渲染模式检查
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

    def _get_obs(self):
        # 返回当前观测值（位置和速度）
        return np.hstack((self.position, self.velocity))

    def _get_info(self):
        return {}  # 暂无额外信息

    def step(self, action):
        # 计算加速度 = 力 / 质量
        acceleration = action / self.mass

        # 更新速度
        self.velocity += acceleration * self.time_step
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed)

        # 更新位置
        self.position += self.velocity * self.time_step

        # 初始步奖励
        reward = self.rewards['step']

        # 处理边界碰撞
        reward = self._handle_boundary_collision(reward)

        # 判断是否达到目标
        terminated, truncated = False, False
        if self._is_goal_reached():
            terminated = True
            reward += self.rewards['goal']

        # 获取观测值并记录轨迹
        obs = self._get_obs()
        info = self._get_info()
        self.trajectory.append(obs.copy())

        return obs, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)  # 初始化随机种子

        # 重置位置、速度、轨迹
        self.position = self.start_position.copy()
        self.velocity = np.zeros(2, dtype=np.float64)
        self.trajectory = []

        return self._get_obs(), self._get_info()

    def _handle_boundary_collision(self, reward):
        # x方向边界碰撞
        if self.position[0] <= 0:
            self.position[0] = 0
            self.velocity[0] *= -1
            reward += self.rewards['bounce']
        elif self.position[0] >= self.width:
            self.position[0] = self.width
            self.velocity[0] *= -1
            reward += self.rewards['bounce']

        # y方向边界碰撞
        if self.position[1] <= 0:
            self.position[1] = 0
            self.velocity[1] *= -1
            reward += self.rewards['bounce']
        elif self.position[1] >= self.height:
            self.position[1] = self.height
            self.velocity[1] *= -1
            reward += self.rewards['bounce']

        return reward

    def _is_goal_reached(self):
        # 如果当前位置距离目标点足够近，则认为完成任务
        distance = np.linalg.norm(self.position - self.target_position)
        return distance < 1.0

    def render(self):
        # 渲染模式判断
        if self.render_mode not in ["rgb_array", "human"]:
            raise ValueError("Unsupported render mode")
        return self._render_frame()

    def _render_frame(self):
        # 创建画布
        canvas = pygame.Surface((self.render_width, self.render_height))
        canvas.fill((255, 255, 255))  # 白色背景

        # 初始化窗口
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_width, self.render_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()

        # 绘制目标点
        target_position_render = self._convert_to_render_coordinate(self.target_position)
        pygame.draw.circle(canvas, (100, 100, 200), target_position_render, 20)

        # 绘制小球当前位置
        ball_position_render = self._convert_to_render_coordinate(self.position)
        pygame.draw.circle(canvas, (0, 0, 255), ball_position_render, 10)

        # 绘制轨迹
        if self.show_epi:
            for i in range(len(self.trajectory) - 1):
                pos1 = self._convert_to_render_coordinate(self.trajectory[i])
                pos2 = self._convert_to_render_coordinate(self.trajectory[i + 1])
                color_val = int(230 * (i / len(self.trajectory)))
                pygame.draw.lines(canvas, (color_val, color_val, color_val), False, [pos1, pos2], width=3)

        # 显示在窗口或返回 ndarray
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.quit()

    def _convert_to_render_coordinate(self, position):
        # 将环境坐标转为渲染窗口中的像素坐标
        return int(position[0] * self.scale), int(self.render_height - position[1] * self.scale)
