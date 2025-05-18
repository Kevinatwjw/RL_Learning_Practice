import random
import gym
import numpy as np
import abc # 用于定义抽象类和抽象方法的模块。

class ReplayBuffer():
    def __init__(self, max_size = 80):
        # 存储状态对的总量
        self.max_size = max_size
        self.buffer = []
    def push_transition(self, transition):
        if transition not in self.buffer: # 这是遍历做法，适合小样本
            self.buffer.append(transition)
            if len(self.buffer) > self.max_size:
                self.buffer = self.buffer[-self.max_size:] # 只保留最新的max_size个状态对(只去掉第一个)
                
    def sample(self, batch_size = 5):
        # 判定batch_size是否大于buffer长度
        if batch_size > self.max_size:
            raise ValueError("采样的长度大于经验回放池大小！")
        # 随机采样
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    def isfull(self):
        """某些强化学习算法（如 off-policy Q-learning）会等 buffer 满了才开始更新；"""
        return len(self.buffer) == self.max_size
    
class Solver():
    def __init__(self,env:gym.Env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None, replay_buffer_size=80):
        # 初始化函数
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # 环境中提取动作空间大小和状态空间大小（要求 env.action_space 和 env.observation_space 必须是 gym.spaces.Discrete 类型）
        self.n_action = env.action_space.n
        self.n_state = env.observation_space.n
        # 初始化Q值表,每个状态-动作对的初始值设为 0（代表“完全不了解环境”）
        self.Q_table = np.zeros((self.n_state, self.n_action),dtype = np.float64)
        # 初始化当前策略的状态值
        self.V_table = np.zeros((self.n_state),dtype = np.float64)
        # 初始时默认每个动作都是最优的(每个状态对应的最优动作列表，而这个列表的长度是不确定的、可能不同的->object)
        self.greedy_policy = np.array([np.arange(self.n_action)] * self.n_state,dtype = object)
        # 标志变量：表示当前 greedy_policy 是否与最新 Q 表匹配(当 Q 值更新后，应将其设为 False)
        self.policy_is_updated = False
        self.rng = np.random.RandomState(1) # 每个智能体独立随机数，不影响全局
        # 设置经验回放池
        self.replay_buffer = ReplayBuffer(replay_buffer_size)
    
    def take_action(self, state):
        """用于epsilon-greedy策略选择动作,该部分属于policy improvement的范畴"""
        # 确保策略已经更新
        if not self.policy_is_updated:
            self.update_policy()
        # epsilon-greedy 策略选择动作
        if np.random.rand() < self.epsilon:
            return self.rng.randint(self.n_action) # 随机选择动作(从 [0, n) 中选一个整数)
        else:
            return self.rng.choice(self.greedy_policy[state]) # 从当前最优动作中随机选择一个，鼓励策略多样性
    
    def update_policy(self):
        """更新当前策略,从Q_table中提取最优动作"""
        # 找出所有最大的Q值对应的动作
        # 这里的np.where()返回的是一个元组，元组中包含了所有最大值的索引[0]将ndarray提取出来
        self.greedy_policy = np.array([np.where(self.Q_table[s] == np.max(self.Q_table[s]))[0] 
                                        for s in range(self.n_state)], dtype=object)
        # 策略更新标志设为 True
        self.policy_is_updated = True
    
    def update_V_table(self):
        """根据当前 Q 表和贪婪策略计算每个状态的状态值函数 V(s)
        若某个状态是接近终点或高奖励区域，那么它的状态值函数 V(s) 会较高；
        若某个状态是接近障碍物或低奖励区域，那么它的状态值函数 V(s) 会较低；
        如果 V 值从左到右、从起点向终点逐渐升高，说明策略在学习从起点走向目标；"""
        # 判断策略是否更新
        if not self.policy_is_updated:
            self.update_policy()
        # 计算每个状态对的状态函数(V(s)=max_a Q(s,a)=E_(a~pi(·|s))[Q(s,a)])
        for s in range(self.n_state):
            self.V_table[s] = self.Q_table[s][self.greedy_policy[s][0]]
            
    @abc.abstractmethod
    def update_Q_table(self):
        """抽象实现"""
        pass

class Sarsa(Solver):
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9,epsilon=0.1,seed=None):
        # 让 Sarsa 自动执行它继承的父类,可以自动执行父类 __init__() 中的所有初始化逻辑。
        super().__init__(env, alpha, gamma, epsilon, seed)
        
    def update_Q_table(self, state, action, reward, next_state, next_action):
        td_target = reward + self.gamma * self.Q_table[next_state, next_action] # 计算 TD 目标
        td_error = td_target - self.Q_table[state, action]
        self.Q_table[state, action] += self.alpha * td_error
        self.policy_is_updated = False # 更新 Q 表后，策略需要更新  

class ExpectedSarsa(Solver):
    def __init__(self, env:gym.Env, alpha=0.1, gamma=0.9,epsilon=0.1,seed=None):
        # 让 Sarsa 自动执行它继承的父类,可以自动执行父类 __init__() 中的所有初始化逻辑。
        super().__init__(env, alpha, gamma, epsilon, seed)
        
    def update_Q_table(self, state, action, reward, next_state, batch_size=0):
        # batch_size = 0为on-policy,否则为off-policy
        if batch_size == 0: 
            Q_Exp = (1-self.epsilon) * self.Q_table[next_state].max() + self.epsilon * self.Q_table[next_state].mean()
            td_target = reward + self.gamma * Q_Exp # 计算 TD 目标
            td_error = td_target - self.Q_table[state, action]
            self.Q_table[state, action] += self.alpha * td_error
        else:
            self.replay_buffer.push_transition(transition=(state, action, reward, next_state))
            transitions = self.replay_buffer.sample(batch_size)
            for s,a,r,s_ in transitions:
                Q_Exp = (1-self.epsilon) * self.Q_table[s_].max() + self.epsilon * self.Q_table[s_].mean()
                td_target = r + self.gamma * Q_Exp # 计算 TD 目标
                td_error = td_target - self.Q_table[s, a]
                self.Q_table[s, a] += self.alpha * td_error
        self.policy_is_updated = False # 更新 Q 表后，策略需要更新  
        
class Nstep_SARSA(Solver):
    """on-policy"""
    def __init__(self, env, alpha=0.1, gamma=0.9, epsilon=0.1, seed=None, nstep=5):
        super().__init__(env, alpha, gamma, epsilon, seed)
        self.nstep = nstep
        self.state_list = []
        self.action_list = []
        self.reward_list = []
    
    def update_Q_table(self, state, action, reward, next_state, next_action, done):
        # 追加当前状态、动作奖励对，注意不用np.append()，因为其每次都会返回一个新的数组，花销大
        self.state_list.append(state)
        self.action_list.append(action)
        self.reward_list.append(reward)
        # 判断是否存储了nstep组数据，N-step SARSA 设计的本质就是基于“完整的n步回报”来更新价值函数。
        if len(self.state_list) == self.nstep:
            # 计算G_{t:t+n}，采用倒序的方法
            G = self.Q_table[next_state,next_action] 
            for i in reversed(range(self.nstep)):
                G = self.gamma * G + self.reward_list[i]       
            td_target = G # TD-target
            # 提取最初的状态动作对，并剔除最老的一组数据
            s_t = self.state_list.pop(0)
            a_t = self.action_list.pop(0)
            self.reward_list.pop(0)
            td_error = td_target - self.Q_table[s_t, a_t]
            self.Q_table[s_t, a_t] += self.alpha * td_error
        # 如果当智能体进入悬崖或者到达终点，此时大概率不满足长度为nstep,需要特殊处理
        if done:
            # 计算G_{t:t+m}，采用倒序的方法
            G = self.Q_table[next_state,next_action] 
            for i in reversed(range(len(self.state_list))):
                G = self.gamma * G + self.reward_list[i]       
                td_target = G # TD-target
                # 提取最初的状态动作对，并剔除最老的一组数据
                s_t = self.state_list.pop(0)
                a_t = self.action_list.pop(0)
                self.reward_list.pop(0)
                td_error = td_target - self.Q_table[s_t, a_t]
                self.Q_table[s_t, a_t] = self.Q_table[s_t, a_t] + self.alpha * td_error
            # 清空列表，开始新的一轮episode
            self.state_list = []
            self.action_list = []
            self.reward_list = []