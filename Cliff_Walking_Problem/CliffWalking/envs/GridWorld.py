import gym
from gym import spaces
import numpy as np
import pygame
"""
这是一个 4 x 12 的网格世界，每一个网格表示一个状态。
1.智能体的起点是左下角的状态，目标是右下角的状态，智能体在每一个状态都可以采取 4 种动作：上、下、左、右。
2.如果智能体采取动作后触碰到边界墙壁则状态不发生改变，否则就会相应到达下一个状态。
3.环境中有一段悬崖，智能体掉入悬崖或到达目标状态都会结束动作并回到起点，也就是说掉入悬崖或者达到目标状态是终止状态。
4.智能体每走一步的奖励是 −1，掉入悬崖的奖励是 −100。
"""
class CliffWalkingEnv(gym.Env):
    """"Cliff Walking Environment"""
    metadata = {
        "render_modes": ["human", "rgb_array"],  
        "render_fps": 30
    }
    def __init__(self, render_mode=None, map_size=(4,12), pix_square_size=20):
        self.nrow = map_size[0]
        self.ncol = map_size[1]
        self.pix_square_size = pix_square_size
        # 定义起始位置
        self.start_location = np.array([0, self.nrow-1],dtype=int)
        self.target_location = np.array([self.ncol-1, self.nrow-1], dtype=int)
        # 定义动作空间
        self.action_space = spaces.Discrete(5) # 让 Gym 和智能体知道这个环境每一步可以选 5 个动作之一
        self._action_to_dir = {
            0: np.array([0, -1]), # 上
            1: np.array([0, 1]), # 下
            2: np.array([-1, 0]), # 左
            3: np.array([1, 0]), # 右
            4: np.array([0, 0]) # 不动
        }
        # 定义观察空间
        self.observation_space = spaces.Dict(
            {   # MultiDiscrete([a, b]) 表示状态是一个二维整数向量
                "agent":spaces.MultiDiscrete([self.ncol, self.nrow]), # 代理的坐标
                "target":spaces.MultiDiscrete([self.ncol, self.nrow]),# 目标的坐标
            }
        )
        # 渲染模式支持‘human’或‘rgb_array’
        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode
        # 渲染窗口的大小渲染模式为 render_mode == 'human' 时用于渲染窗口的组件
        self.window  = None # 两个变量是为了图形渲染做准备
        self.clock = None
    
    """因为在 env.reset() 和 env.step() 方法中都要返回 observation，
    可以设置一个内部方法进行 state 到 observation 的转换。
    另外，这里将二者返回的附加信息 info 定义为 agent 当前状态距离目标位置的曼哈顿距离"""    
    def _get_obs(self):
        """获取当前状态"""
        # 观察空间定义为 agent 和 target 的坐标
        return {
                "agent": self.agent_location,
                "target": self.target_location
            }
    def _get_info(self):
        """获取额外信息"""
         # 附加信息定义为 agent 当前位置到 target 的曼哈顿距离
        return {
                "distance": np.linalg.norm(self.agent_location - self.target_location,ord = 1)
            }
    
    """reset 方法用于启动一个新轨迹的交互过程，可以假定在调用 reset() 之前不会调用 step() 方法，
    同时，当 step() 方法发出 terminated 或 truncated 信号时应该调用 reset()
    reset() 方法应该返回初始 observation 和一些辅助 info，可以使用之前实现的 _get_obs 和 _get_info 方法"""
    def reset(self, seed=None, options=None):
        """
        reset() 是 Gym 强制要求你实现的方法之一。
        seed: 传入的随机种子，用于确保复现性（Gym ≥ 0.21 引入）
        options: 可选参数，允许用户自定义 reset 逻辑中的额外参数（一般可以忽略）
        """
        super().reset(seed=seed) # 调用父类的 reset 方法不要写成self.seed(seed) 否则会报错
        # 设置初始状态
        self.agent_location = self.start_location.copy() # （0, 3）
        # 目标位置
        self.target_location = self.target_location.copy() # （11, 3）
        observation = self._get_obs() # 获取当前状态
        info = self._get_info() # 获取额外信息
        # 可以在此刷新渲染，但本例需要渲染最新策略，所以在测试时更新策略后再手动调用 render 方法(但通常我们在策略模型完成一步 action 后才渲染)
        # if self.render_mode == "human":	
        #    self._render_frame()
        return observation, info

    """step() 方法通常包含环境的大部分逻辑。
    它接受一个 action，计算应用该 action 后的环境 state，并返回元组 (observation, reward, terminated, truncated, info)
    对于 observation 和 info 直接使用之前实现的 _get_obs 和 _get_info 方法得到，reward 根据环境定义设置、
    terminated信号设为 “达到目标位置”，truncated信号设为 “落下悬崖”
    """
    def step(self, action):
        """参数 action：是一个整数，取值范围 [0, 4]，表示 agent 当前选择的动作（上下左右或不动）。
        函数的目标是：根据当前状态和动作，更新 agent 位置，判断奖励与终止状态，并返回结果。
        """
        # 状态转移：执行动作
        self.agent_location, out_of_bounds = self._state_transition(self.agent_location, action)
        # 判断是否落下悬崖(truncated)或者到达目标位置（terminated）
        # 定义两者位置(bool)
        terminated = np.array_equal(self.agent_location, self.target_location) # 到达目标位置
        truncated = self.agent_location[1].item() == self.nrow - 1 and self.agent_location[0].item() not in [0,self.ncol-1] 
        # 定义奖励
        if truncated:
            reward = -100 # 掉入悬崖的奖励是 −100
        elif terminated:
            reward = 0 # 到达目标位置的奖励是 0
        elif out_of_bounds:
            reward = -10 # 越界的奖励是 −10
        else:
            reward = -1 # 每走一步的奖励是 −1
        # 获取当前状态信息
        observation = self._get_obs()
        info = self._get_info()
        # 可以在这里刷新渲染，但我这里需要渲染最新策略，所以在测试时再手动调用 render 方法
        #if self.render_mode == "human":
        #    self._render_frame()
        return observation, reward, terminated,truncated, info
        
    def _state_transition(self, state, action):
        """状态转移函数
        state: 当前状态(type: np.array)
        action: 当前动作(type: int)
        返回：下一个位置坐标
        """
        direction = self._action_to_dir[action] # 获取动作对应的方向
        next_state = state + direction
        # 裁剪法保证下一个状态在边界内
        clip_next_state = np.array([
            np.clip(next_state[0], 0, self.ncol-1).item(),
            np.clip(next_state[1], 0,self.nrow-1).item()
        ]) 
        out_of_bounds = not np.array_equal(next_state, clip_next_state) # 判断是否越界
        return clip_next_state, out_of_bounds

    def render(self, state_values=None, policy=None):
        """
        Gym 要求环境类必须实现 render() 方法，用于状态可视化
        这里支持两个可选输入参数：
        state_values: 状态价值（用于上色）(type: np.ndarray)
        policy: 策略（用于画箭头）(type: np.ndarray of list[int])
        """
        if self.render_mode == "rgb_array":
            return self._render_frame()
        elif self.render_mode == "human":
            return self._render_frame(state_values, policy)
        else: 
            raise ValueError(f"Unsupported render_mode: {self.render_mode}")

    def _render_frame(self, state_values=None, policy=None):
        """
        渲染环境当前帧的画面，可用于人类观察（human 模式）或图像提取（rgb_array 模式）。
        在 human 模式下，该函数将在屏幕上绘制：
            - 环境的网格结构，包括非悬崖区域和悬崖区域；
            - 每个状态格的值函数（value），使用颜色深浅表示；
            - 当前智能体的位置（蓝色圆点）；
            - 当前策略的贪婪动作（绿色虚线箭头）。
        在 rgb_array 模式下，该函数返回一帧 RGB 图像（不含 value 与策略箭头），用于图像型观测输入（如 CNN）。
        参数:
        state_values : np.ndarray, optional
            状态价值函数，shape 通常为 (nrow, ncol)。每个格子的颜色深浅将根据该值渲染。
            仅在 render_mode 为 "human" 时生效。
        policy : np.ndarray of list[int], optional
            策略表，shape 为 (nrow * ncol, )，每个元素是当前状态下的最优动作列表（支持贪婪策略并列动作）。
            每个动作用绿色虚线箭头从当前状态格指向下一个状态格。仅在 render_mode 为 "human" 时生效。
        返回值:
        None 或 np.ndarray
            - 若 render_mode 为 "human"，则在屏幕上显示画面，函数无返回值；
            - 若 render_mode 为 "rgb_array"，则返回一帧 RGB 图像，shape 为 (height, width, 3)，类型为 np.uint8。
        注意事项:
        - 起点（左下角）使用灰色填充；
        - 终点（右下角）使用绿色填充；
        - 悬崖区域（最底行中间）使用黑色表示；
        - 为避免 CNN 提取观测被干扰，在 rgb_array 模式中不会绘制状态值或策略箭头。
        """
        pix_square_size = self.pix_square_size
        # 创建画布，用于绘制当前整张网络图
        canvas = pygame.Surface((self.ncol * pix_square_size, self.nrow * pix_square_size))
        canvas.fill((255, 255, 255)) # 初始背景为白色
        # 初始化窗口和时钟（只在“human”模式下使用）
        if self.render_mode == "human":
            pygame.init() # 初始化 pygame
            pygame.display.init() # 初始化显示窗口
            self.window = pygame.display.set_mode((self.ncol * pix_square_size, self.nrow * pix_square_size)) # 创建窗口
        if self.clock is None and self.render_mode == "human":
            self.clock = pygame.time.Clock() # 创建时钟
        # 初始化背景为白色画布(作用是清除之前的画布，准备好白色背景)
        pygame.draw.rect(
            canvas,
            (255,255,255),
            pygame.Rect((0,0), # 起始原点在画布左上角
                        (self.ncol * pix_square_size, self.nrow * pix_square_size) # 画布大小
                        ),
            )
        # 绘制前nrow-1行（非悬崖）方格
        if self.render_mode == "human" and isinstance(state_values, np.ndarray): # 如果是"human"并且有状态值
            for col in range(self.ncol):
                for row in range(self.nrow-1):
                    # 获取当前格的state_value:
                    state_value = state_values[row][col].item()
                    max_state_value = 1 if np.abs(state_values).max() ==0 else np.abs(state_values).max()
                    #计算颜色深浅
                    pygame.draw.rect(
                        canvas,
                        (abs(state_value)/max_state_value*255,20,20), # 颜色深浅
                        pygame.Rect(
                            (col * pix_square_size, row * pix_square_size), # 该状态的像素起始坐标
                            (pix_square_size-1, pix_square_size-1) # # -1 使格子之间有清晰边界线
                        ),
                    )
        else: # 如果是"rgb_array"模式不渲染状态颜色
            for col in range(self.ncol):
                for row in range(self.nrow-1):
                    pygame.draw.rect(
                        canvas,
                        (150,150,150), # 灰色
                        pygame.Rect(
                            (col * pix_square_size, row * pix_square_size), # 该状态的像素起始坐标
                            (pix_square_size-1, pix_square_size-1) # # -1 使格子之间有清晰边界线
                        ),
                    )
        # 绘制悬崖区域
        for col in range(self.ncol):
            if col == 0:
                color = (100,100,100) # 起点
            elif col == self.ncol-1:
                color = (100,150,100) # 终点
            else:
                color = (0,0,0) # 悬崖区域
            pygame.draw.rect(
                canvas,
                color,
                pygame.Rect(
                    (col * pix_square_size, (self.nrow-1) * pix_square_size), # 该状态的像素起始坐标
                    (pix_square_size-1, pix_square_size-1) # # -1 使格子之间有清晰边界线
                ),
            )
        # 绘制智能体位置
        pygame.draw.circle(
            canvas,
            (0,0,255), # 蓝色
            (self.agent_location+0.5) * pix_square_size, # 圆心坐标
            pix_square_size/3, # 圆的半径是格子边长的 1/3，蓝色代表当前状态
        )
        # human 模式下渲染基于 Q value 的贪心策略
        # 如果当前渲染模式为 'human' 且传入了 policy 数组（策略表），则可视化策略
        if self.render_mode == "human" and isinstance(policy, np.ndarray):
            for col in range(self.ncol):
                for row in range(self.nrow-1):
                    # 将二维网络坐标用映射为一维索引（按列优先展开（0,0）、（0,1）...）
                    hash_position = col * self.nrow + row
                    actions = policy[hash_position] # 获取当前状态的动作
                    for a in actions:
                        next_s, _ = self._state_transition(np.array([col, row]), a)
                        if not np.array_equal(next_s, np.array([col, row])):
                            start = np.array([col*pix_square_size+0.5*pix_square_size,row*pix_square_size+0.5*pix_square_size])
                            end = next_s * pix_square_size + 0.5 * pix_square_size
                            dot_num = 15
                            for i in range(dot_num):
                                pygame.draw.rect(
                                    canvas,
                                    (10, 255-i*175/dot_num, 10),
                                    pygame.Rect(
                                        start + (end-start) * i/dot_num,
                                        (2,2)
                                    ),
                                )
            # 最后一行只绘制起点策略
            col, row = 0, self.nrow-1
            hash_position = col*self.nrow + row
            actions = policy[hash_position]
            for a in actions:
                next_s, _ = self._state_transition(np.array([col,row]), a)
                if (next_s != np.array([col,row])).sum() != 0:
                    start = np.array([col*pix_square_size+0.5*pix_square_size,row*pix_square_size+0.5*pix_square_size])
                    end = next_s*pix_square_size+0.5*pix_square_size
                    dot_num = 15
                    for i in range(dot_num):
                        pygame.draw.rect(
                            canvas,
                            (10, 255-i*175/dot_num, 10),
                            pygame.Rect(
                                start + (end-start) * i/dot_num,
                                (2,2)
                            ),
                        )
        # 'human'模式下显示窗口
        if self.render_mode == "human":
            # 将画布复制到窗口中
            self.window.blit(canvas,canvas.get_rect()) # canvas.get_rect() 表示整个画布区域。
            pygame.event.pump() # 刷新事件队列，让 Pygame 维持窗口响应
            pygame.display.update() # 更新窗口显示
            # 控制帧率
            self.clock.tick(self.metadata["render_fps"]) # 设置帧率
        # 'rgb_array'染模式下画面会转换为像素 ndarray 形式返回，适用于用 CNN 进行状态观测的情况，为避免影响观测不要渲染价值颜色和策略
        else:
            # np.transpose置轴顺序，将(W, H, 3)改为 (H, W, 3) 格式
             return np.transpose(np.array(pygame.surfarray.pixels3d(canvas)), axes=(1, 0, 2))
    def close(self):
        """
        关闭环境所有使用的开放资源
        """
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()