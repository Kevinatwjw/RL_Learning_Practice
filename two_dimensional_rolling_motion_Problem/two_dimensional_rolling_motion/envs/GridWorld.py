import gym
from gym import spaces
import numpy as np
import pygame
import time

class RollingBall(gym.Env):
    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 500,
    }

    def __init__(self, render_mode="human", width=10, height=10, show_epi=False):
        self.max_speed = 5.0
        self.width = width
        self.height = height
        self.show_epi = show_epi

        # 定义动作空间和观测空间为 float32
        self.action_space = spaces.Box(low=-0.1, high=0.1, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([0.0, 0.0, -self.max_speed, -self.max_speed], dtype=np.float32),
            high=np.array([width, height, self.max_speed, self.max_speed], dtype=np.float32),
            dtype=np.float32
        )

        # 初始化速度和位置为 float32
        self.velocity = np.zeros(2, dtype=np.float32)
        self.mass = 0.05
        self.time_step = 0.01
        self.friction_coeff = 0.0046
        self.restitution_coeff = 0.8

        # 奖励设计
        self.rewards = {
            'step': -2.0,    # 每步惩罚
            'bounce': -10.0, # 碰撞惩罚
            'goal': 300.0    # 目标奖励
        }

        # 目标和起点位置为 float32
        self.target_position = np.array([self.width * 0.8, self.height * 0.8], dtype=np.float32)
        self.start_position = np.array([width * 0.2, height * 0.2], dtype=np.float32)
        self.position = self.start_position.copy()

        # 渲染参数
        self.render_width = 300
        self.render_height = 300
        self.scale = self.render_width / self.width
        self.window = None
        self.clock = None
        self.trajectory = []

        # 渲染模式检查
        assert render_mode is None or render_mode in self.metadata["render_modes"], \
            f"Invalid render_mode: {render_mode}. Supported modes: {self.metadata['render_modes']}"
        self.render_mode = render_mode

    def _get_obs(self):
        # 返回当前观测值（位置和速度），确保 float32
        return np.hstack((self.position, self.velocity)).astype(np.float32)

    def _get_info(self):
        # 返回额外信息，便于调试
        return {'distance_to_target': np.linalg.norm(self.position - self.target_position)}

    def step(self, action):
        # 确保动作是 float32
        action = np.array(action, dtype=np.float32)
        # 计算加速度，包含摩擦力
        acceleration = action / self.mass - (self.friction_coeff / self.mass) * self.velocity
        # 更新速度
        self.velocity += acceleration * self.time_step
        self.velocity = np.clip(self.velocity, -self.max_speed, self.max_speed).astype(np.float32)
        # 更新位置
        self.position += self.velocity * self.time_step
        # 计算距离目标的奖励
        distance = np.linalg.norm(self.position - self.target_position)
        reward = self.rewards['step'] - 0.1 * distance  # 添加密集奖励
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
        super().reset(seed=seed)
        # 重置位置、速度、轨迹
        self.position = self.start_position.copy()
        self.velocity = np.zeros(2, dtype=np.float32)
        self.trajectory = []
        return self._get_obs(), self._get_info()

    def _handle_boundary_collision(self, reward):
        # x方向边界碰撞
        if self.position[0] <= 0:
            self.position[0] = 0
            self.velocity[0] *= -self.restitution_coeff
            reward += self.rewards['bounce']
        elif self.position[0] >= self.width:
            self.position[0] = self.width
            self.velocity[0] *= -self.restitution_coeff
            reward += self.rewards['bounce']
        # y方向边界碰撞
        if self.position[1] <= 0:
            self.position[1] = 0
            self.velocity[1] *= -self.restitution_coeff
            reward += self.rewards['bounce']
        elif self.position[1] >= self.height:
            self.position[1] = self.height
            self.velocity[1] *= -self.restitution_coeff
            reward += self.rewards['bounce']
        # 确保 position 和 velocity 是 float32
        self.position = self.position.astype(np.float32)
        self.velocity = self.velocity.astype(np.float32)
        return reward

    def _is_goal_reached(self):
        # 判断是否接近目标
        distance = np.linalg.norm(self.position - self.target_position)
        return distance < 1.0

    def render(self):
        if self.render_mode not in ["rgb_array", "human"]:
            raise ValueError(f"不支持的渲染模式: {self.render_mode}")
        return self._render_frame()

    def _render_frame(self):
        canvas = pygame.Surface((self.render_width, self.render_height))
        canvas.fill((255, 255, 255))  # 白色背景
        if self.window is None and self.render_mode == "human":
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode((self.render_width, self.render_height))
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock()
        # 绘制目标点
        target_position_render = self._convert_to_render_coordinate(self.target_position)
        pygame.draw.circle(canvas, (100, 100, 200), target_position_render, 20)
        # 绘制小球
        ball_position_render = self._convert_to_render_coordinate(self.position)
        pygame.draw.circle(canvas, (0, 0, 255), ball_position_render, 10)
        # 绘制轨迹
        if self.show_epi:
            for i in range(len(self.trajectory) - 1):
                pos1 = self._convert_to_render_coordinate(self.trajectory[i])
                pos2 = self._convert_to_render_coordinate(self.trajectory[i + 1])
                color_val = int(230 * (i / len(self.trajectory)))
                pygame.draw.lines(canvas, (color_val, color_val, color_val), False, [pos1, pos2], width=3)
        if self.render_mode == "human":
            self.window.blit(canvas, canvas.get_rect())
            pygame.event.pump()
            pygame.display.update()
            self.clock.tick(self.metadata["render_fps"])
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()
            self.window = None

    def _convert_to_render_coordinate(self, position):
        return int(position[0] * self.scale), int(self.render_height - position[1] * self.scale)