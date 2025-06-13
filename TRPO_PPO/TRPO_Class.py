import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
import numpy as np
import copy

class PolicyNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x=F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)
    
class PolicyNet_Continuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim):
        super(PolicyNet_Continuous, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc_mu = torch.nn.Linear(hidden_dim, action_dim)
        self.fc_std = torch.nn.Linear(hidden_dim, action_dim)
    
    def forward(self, x):
        x=F.relu(self.fc1(x))
        # 计算均值Mu,因为倒立摆环境下，其输出范围为[-2,2]，所以需要将输出范围映射到[-2,2]
        mu = 2 * torch.tanh(self.fc_mu(x))
        # 计算标准差Std,因为倒立摆环境下，其输出范围为[-2,2]，所以需要将输出范围映射到[-2,2]
        std = F.softplus(self.fc_std(x))
        return mu, std
        
class ValueNet(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim):
        super(ValueNet, self).__init__()
        self.fc1 = torch.nn.Linear(state_dim, hidden_dim)
        self.fc2 = torch.nn.Linear(hidden_dim, 1)
    
    def forward(self, x):
        x=F.relu(self.fc1(x))
        return self.fc2(x)
def compute_advantage(gamma, lmbda, td_error):
    """计算优势函数（TRPO）,公式：A_t = 求和：gamma * lmbda * delta_t

    Args:
        gamma (_type_): 折扣系数
        lmbda (_type_): 优势函数的衰减系数
        td_error (_type_): 时序差分误差

    Returns:
        _type_: 优势函数
    """
    # .numpy()将tensor数组转化为numpy数组
    td_error = td_error.detach().numpy()
    advantage_list = []
    advantage = 0.0
    # 从前往后遍历td_error，其中td_error的形状是(batch_size,)，[::-1]表示从后往前遍历
    for delta in td_error[::-1]:
        # 计算优势函数，公式：A_t = gamma * lmbda * A_t+1 + delta_t
        advantage = gamma * lmbda * advantage + delta
        advantage_list.append(advantage)
    advantage_list.reverse()
    return torch.tensor(advantage_list, dtype=torch.float)   
          
class TRPO(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lmbda, KL_constraint, alpha, critic_lr, gamma, device):
        super(TRPO, self).__init__()
        self.lmbda = lmbda
        self.KL_constraint = KL_constraint
        self.alpha = alpha
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device
        self.actor = PolicyNet(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        # 注意：这里不要用state.unsqueeze(0)，因为state此时形状已经是（1，state_dim）
        # 如果用state.unsqueeze(0)，则state的形状会变成（1，1，state_dim），
        # 这样会导致probs的形状变成（1，1，action_dim），
        # 而action_dist.sample()的形状是（1，），
        # 这样会导致action的形状是（1，），
        probs = self.actor(state)
        action_dist = torch.distributions.Categorical(probs)
        action = action_dist.sample()
        return action.item()
    
    def hessian_matrix_vector_product(self, states, old_action_dicts, vector):
        """计算hessian矩阵与向量vector的乘积
        Args:
            states: 状态
            old_action_dicts: 旧的动作分布
            vector: 向量
        Returns:
            hessian矩阵与向量vector的乘积
        """
        # 计算hessian矩阵与向量vector的乘积
        new_action_dicts = torch.distributions.Categorical(self.actor(states))
        # 计算新的动作和旧的动作之间的KL散度
        # 计算KL散度的公式：D_KL(pi_old||pi_new) = sum(pi_old * log(pi_old/pi_new))
        # 这里使用了平均Kl散度，因为在实际实现中，无法直接计算整个状态空间 $\rho_{\theta_{\text{old}}}(s)$ 上的期望
        # 平均KL散度是期望的的蒙特卡洛估计，确保hessian矩阵可以反应所有状态的分布变化
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dicts, new_action_dicts))
        # 计算kl散度的梯度，create_graph=True表示创建计算图，允许后续的计算可以继续求导
        # autograd.grad()函数用于计算函数kl的梯度，返回值是一个包含梯度的元组,参数self.actor.parameters()表示kl是关于actor的函数
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        # 将kl散度的梯度与向量vector相乘，并转换为向量，kl_grad_vector形状是(params_num,)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL散度的梯度先和向量进行点积,dot用法：torch.dot(a, b) = sum(a * b)
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # 计算hessian矩阵
        kl_grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        # 将kl_grad2转换为向量，kl_grad2_vector形状是(params_num,)
        kl_grad2_vector = torch.cat([grad.view(-1) for grad in kl_grad2])
        return kl_grad2_vector
    
    def conjugate_gradient(self, grad, states, old_action_dicts):
        """
        使用共轭梯度法 (Conjugate Gradient Method) 求解线性方程 Hx = grad，
        近似计算自然梯度方向 H^-1 * grad，用于 TRPO 的优化。

        参数:
        - grad (torch.Tensor): 目标梯度向量 g，维度与参数向量一致
        - states (torch.Tensor): 状态张量，用于计算 Hessian-向量乘积
        - old_action_dists (torch.distributions.Categorical): 旧策略动作分布

        返回:
        - x (torch.Tensor): 近似解 H^-1 * grad，维度与 grad 一致
        """
        # 初始化x,r,p，x是解，r是残差，p是搜索方向
        x = torch.zeros_like(grad)
        # r 是残差向量，初始时 r_0 = g - H * x_0，x_0 = 0 故 r_0 = g。
        r = grad.clone()
        # p 是共轭方向向量，初始时 p_0 = r_0 = g。
        p = grad.clone()
        # 初始残差的内积r^T * r,是残差的平方范数，用于计算步长 alpha。
        r_dot_r = torch.dot(r, r)
        # 主循环，次数暂定为10次，视问题复杂度而定
        for i in range(10):
            # 计算Hessian矩阵的向量乘积H*p,p是共轭方向向量,用p近似H^-1 * grad
            Hp = self.hessian_matrix_vector_product(states, old_action_dicts, p)
            # 计算步长alpha，alpha = r_dot_r / (p^T * H * p),最小化目标函数沿着p_k方向的增量
            alpha = r_dot_r / torch.dot(p, Hp)
            # 更新解向量，x = x + alpha * p
            x += alpha * p
            # 更新残差,公式：r_k+1 = r_k - alpha * H * p_k
            r -= alpha * Hp
            # 如果残差r的范数小于1e-10，则认为收敛，退出循环
            new_r_dot_r = torch.dot(r,r)
            if new_r_dot_r < 1e-10:
                break
            # 计算共轭方向系数 beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
            beta = new_r_dot_r / r_dot_r
            # 更新共轭方向向量，p = r + beta * p
            p = r + beta * p
            r_dot_r = new_r_dot_r
        return x
    
    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """
        计算 surrogate 目标函数
        计算 TRPO 的代理目标函数 (Surrogate Objective)，用于评估策略更新的效果。
        代理目标衡量新策略相对于旧策略的期望优势。
        参数:
        - states (torch.Tensor): 状态张量，形状为 [batch_size, state_dim]
        - actions (torch.Tensor): 动作张量，形状为 [batch_size, 1]
        - advantage (torch.Tensor): 优势函数值，形状为 [batch_size, 1]
        - old_log_probs (torch.Tensor): 旧策略的对数概率，形状为 [batch_size, 1]
        - actor (PolicyNet): 策略网络实例 (可以是当前或新策略)
        返回:
        - torch.Tensor: 代理目标函数的均值，标量
        """
        # 计算状态下采取的动作概率对数
        log_probs = torch.log(actor(states).gather(1, actions))
        # 计算代理目标函数，公式：L_t(theta) = E[A_t * pi_theta(a_t|s_t) / pi_theta_old(a_t|s_t)]
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)
    
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dicts, max_vec):
        """
        执行线性搜索以找到满足 KL 散度约束和代理目标增量的最优参数更新。
        线性搜索通过逐步减小步长，验证新策略是否改善性能且符合约束。

        参数:
        - states (torch.Tensor): 状态张量，形状为 [batch_size, state_dim]
        - actions (torch.Tensor): 动作张量，形状为 [batch_size, 1]
        - advantage (torch.Tensor): 优势函数值，形状为 [batch_size, 1]
        - old_log_probs (torch.Tensor): 旧策略的对数概率，形状为 [batch_size, 1]
        - old_action_dists (torch.distributions.Categorical): 旧策略动作分布
        - max_vec (torch.Tensor): 最大更新方向向量 (通常是自然梯度方向)

        返回:
        - new_para (torch.Tensor): 优化后的参数向量，或返回 old_para 若未找到合适解
        """
        # 将当前策略网络的参数展平为向量，作为初始向量(包含所有权重和偏置)
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        # 计算当前的代理目标值，作为基准(其代表就策略的代理目标的值)
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):
            # 当前步长的减小，确保满足约束的解
            coef = self.alpha ** i
            # 计算新的参数参量,公式：new_para ≈ θ_old + α^i * H^-1 * g，是参数的更新尝试
            new_para = old_para + coef * max_vec
            # 创建新的策略网络副本，采用copy.deepcopy()避免修改原网络
            new_actor = copy.deepcopy(self.actor)
            # 将新的参数给副本网络
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            # 计算新策略的动作的概率分布
            new_action_dicts = torch.distributions.Categorical(new_actor(states))
            # 计算新策略与旧策略之间的KL散度
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dicts, new_action_dicts))
            # 计算新的策略的代理目标的值
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            # 判断新的参数下是否满足约束
            if kl_div < self.KL_constraint and new_obj > old_obj:
                return new_para
        return old_para
    
    def policy_update(self, states, actions, old_action_dicts, old_log_probs, advantage):
        """
        更新策略网络参数，使用 TRPO 算法优化代理目标。
        该方法结合共轭梯度法和线性搜索，确保策略改进并满足 KL 散度约束。

        参数:
        - states (torch.Tensor): 状态张量，形状为 [batch_size, state_dim]
        - actions (torch.Tensor): 动作张量，形状为 [batch_size, 1]
        - old_action_dists (torch.distributions.Categorical): 旧策略动作分布
        - old_log_probs (torch.Tensor): 旧策略的对数概率，形状为 [batch_size, 1]
        - advantage (torch.Tensor): 优势函数值，形状为 [batch_size, 1]

        无返回值: 直接更新 self.actor 的参数
        """
        # 计算当前的代理目标值，用于后续的优化基准
        surrogate = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        # 计算代理目标函数的梯度
        grads = torch.autograd.grad(surrogate, self.actor.parameters())
        # 将梯度展平为一个向量，并分离计算图避免梯度积累,obj_grad = g，是展平后的梯度向量 g = ∇_θ L_θ_old(θ)|_θ=θ_old。detach() 防止梯度回溯，提高效率。
        surrogate_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 使用共轭梯度法计算自然梯度方向 x = H^-1 * g,descent_direction ≈ H^-1 * g，其中 H 是 Fisher 信息矩阵，
        # g 是梯度向量。共轭梯度法高效求解 Hx = g，x 是自然梯度方向。
        descent_direction = self.conjugate_gradient(surrogate_grad, states, old_action_dicts)
        # 计算 Hessian-向量乘积 H * descent_direction，Hd = H * (H^-1 * g)，验证共轭梯度法的中间结果。
        Hd = self.hessian_matrix_vector_product(states, old_action_dicts, descent_direction)
        # 计算最大步长系数，确保KL散度约束条件，max_coef = sqrt(2δ / (g^T H^-1 g))，δ 是 KL 约束阈值。
        max_coef = torch.sqrt(2 * self.KL_constraint / (torch.dot(descent_direction, Hd)+1e-8))
        # 执行线性搜索，找到满足步长的最大优化参数,new_para = θ_old + α * H^-1 * g，α 从 max_coef 逐步减小。
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dicts, descent_direction * max_coef)
        # 将优化好的参数加载到策略网络
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.long).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_error = td_target - self.critic(states)
        # 计算优势函数,td_error在GPU中，外部计算优势函数以numpy为核心，在cpu上
        advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)
        # 计算旧的对数策略分布概率
        old_log_probs = torch.log(self.actor(states).gather(1,actions)).detach()
        # 计算旧的动作分布
        old_action_dicts = torch.distributions.Categorical(self.actor(states).detach())
        # 更新Critic网络参数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 更新策略网络
        self.policy_update(states, actions, old_action_dicts, old_log_probs, advantage)
        
        
        
class TRPO_Continuous(torch.nn.Module):
    def __init__(self, state_dim, hidden_dim, action_dim, lmbda, KL_constraint, alpha, critic_lr, gamma, device):
        super(TRPO_Continuous, self).__init__()
        self.lmbda = lmbda
        self.KL_constraint = KL_constraint
        self.alpha = alpha
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.device = device
        self.actor = PolicyNet_Continuous(state_dim, hidden_dim, action_dim).to(self.device)
        self.critic = ValueNet(state_dim, hidden_dim).to(self.device)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=self.critic_lr)
    
    def take_action(self, state):
        state = torch.tensor([state], dtype=torch.float).to(self.device)
        mu, std = self.actor(state)
        action_dist = torch.distributions.Normal(mu, std)
        action = action_dist.sample()
        return [action.item()]
    
    def hessian_matrix_vector_product(self, states, old_action_dicts, vector, damping=1):
        """计算hessian矩阵与向量vector的乘积
        Args:
            states: 状态
            old_action_dicts: 旧的动作分布
            vector: 向量
        Returns:
            hessian矩阵与向量vector的乘积
        """
        # 计算hessian矩阵与向量vector的乘积
        mu, std = self.actor(states)
        new_action_dicts = torch.distributions.Normal(mu, std)
        # 计算新的动作和旧的动作之间的KL散度
        # 计算KL散度的公式：D_KL(pi_old||pi_new) = sum(pi_old * log(pi_old/pi_new))
        # 这里使用了平均Kl散度，因为在实际实现中，无法直接计算整个状态空间 $\rho_{\theta_{\text{old}}}(s)$ 上的期望
        # 平均KL散度是期望的的蒙特卡洛估计，确保hessian矩阵可以反应所有状态的分布变化
        kl = torch.mean(torch.distributions.kl.kl_divergence(old_action_dicts, new_action_dicts))
        # 计算kl散度的梯度，create_graph=True表示创建计算图，允许后续的计算可以继续求导
        # autograd.grad()函数用于计算函数kl的梯度，返回值是一个包含梯度的元组,参数self.actor.parameters()表示kl是关于actor的函数
        kl_grad = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
        # 将kl散度的梯度与向量vector相乘，并转换为向量，kl_grad_vector形状是(params_num,)
        kl_grad_vector = torch.cat([grad.view(-1) for grad in kl_grad])
        # KL散度的梯度先和向量进行点积,dot用法：torch.dot(a, b) = sum(a * b)
        kl_grad_vector_product = torch.dot(kl_grad_vector, vector)
        # 计算hessian矩阵
        kl_grad2 = torch.autograd.grad(kl_grad_vector_product, self.actor.parameters())
        # 将kl_grad2转换为向量，kl_grad2_vector形状是(params_num,)
        kl_grad2_vector = torch.cat([grad.view(-1) for grad in kl_grad2])
        # 添加相应的阻尼项，防止Hessian矩阵奇异，保证其正定性
        return kl_grad2_vector + damping * vector
    
    def conjugate_gradient(self, grad, states, old_action_dicts):
        """
        使用共轭梯度法 (Conjugate Gradient Method) 求解线性方程 Hx = grad，
        近似计算自然梯度方向 H^-1 * grad，用于 TRPO 的优化。

        参数:
        - grad (torch.Tensor): 目标梯度向量 g，维度与参数向量一致
        - states (torch.Tensor): 状态张量，用于计算 Hessian-向量乘积
        - old_action_dists (torch.distributions.Categorical): 旧策略动作分布

        返回:
        - x (torch.Tensor): 近似解 H^-1 * grad，维度与 grad 一致
        """
        # 初始化x,r,p，x是解，r是残差，p是搜索方向
        x = torch.zeros_like(grad)
        # r 是残差向量，初始时 r_0 = g - H * x_0，x_0 = 0 故 r_0 = g。
        r = grad.clone()
        # p 是共轭方向向量，初始时 p_0 = r_0 = g。
        p = grad.clone()
        # 初始残差的内积r^T * r,是残差的平方范数，用于计算步长 alpha。
        r_dot_r = torch.dot(r, r)
        # 主循环，次数暂定为10次，视问题复杂度而定
        for i in range(10):
            # 计算Hessian矩阵的向量乘积H*p,p是共轭方向向量,用p近似H^-1 * grad
            Hp = self.hessian_matrix_vector_product(states, old_action_dicts, p)
            # 计算步长alpha，alpha = r_dot_r / (p^T * H * p),最小化目标函数沿着p_k方向的增量
            alpha = r_dot_r / torch.dot(p, Hp)
            # 更新解向量，x = x + alpha * p
            x += alpha * p
            # 更新残差,公式：r_k+1 = r_k - alpha * H * p_k
            r -= alpha * Hp
            # 如果残差r的范数小于1e-10，则认为收敛，退出循环
            new_r_dot_r = torch.dot(r,r)
            if new_r_dot_r < 1e-10:
                break
            # 计算共轭方向系数 beta = (r_{k+1}^T * r_{k+1}) / (r_k^T * r_k)
            beta = new_r_dot_r / r_dot_r
            # 更新共轭方向向量，p = r + beta * p
            p = r + beta * p
            r_dot_r = new_r_dot_r
        return x
    
    def compute_surrogate_obj(self, states, actions, advantage, old_log_probs, actor):
        """
        计算 surrogate 目标函数
        计算 TRPO 的代理目标函数 (Surrogate Objective)，用于评估策略更新的效果。
        代理目标衡量新策略相对于旧策略的期望优势。
        参数:
        - states (torch.Tensor): 状态张量，形状为 [batch_size, state_dim]
        - actions (torch.Tensor): 动作张量，形状为 [batch_size, 1]
        - advantage (torch.Tensor): 优势函数值，形状为 [batch_size, 1]
        - old_log_probs (torch.Tensor): 旧策略的对数概率，形状为 [batch_size, 1]
        - actor (PolicyNet): 策略网络实例 (可以是当前或新策略)
        返回:
        - torch.Tensor: 代理目标函数的均值，标量
        """
        mu, std = actor(states)
        action_dists = torch.distributions.Normal(mu, std)
        log_probs = action_dists.log_prob(actions)
        ratio = torch.exp(log_probs - old_log_probs)
        return torch.mean(ratio * advantage)
        
    
    def line_search(self, states, actions, advantage, old_log_probs, old_action_dicts, max_vec):
        """
        执行线性搜索以找到满足 KL 散度约束和代理目标增量的最优参数更新。
        线性搜索通过逐步减小步长，验证新策略是否改善性能且符合约束。

        参数:
        - states (torch.Tensor): 状态张量，形状为 [batch_size, state_dim]
        - actions (torch.Tensor): 动作张量，形状为 [batch_size, 1]
        - advantage (torch.Tensor): 优势函数值，形状为 [batch_size, 1]
        - old_log_probs (torch.Tensor): 旧策略的对数概率，形状为 [batch_size, 1]
        - old_action_dists (torch.distributions.Categorical): 旧策略动作分布
        - max_vec (torch.Tensor): 最大更新方向向量 (通常是自然梯度方向)

        返回:
        - new_para (torch.Tensor): 优化后的参数向量，或返回 old_para 若未找到合适解
        """
        # 将当前策略网络的参数展平为向量，作为初始向量(包含所有权重和偏置)
        old_para = torch.nn.utils.convert_parameters.parameters_to_vector(self.actor.parameters())
        # 计算当前的代理目标值，作为基准(其代表就策略的代理目标的值)
        old_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        for i in range(15):
            # 当前步长的减小，确保满足约束的解
            coef = self.alpha ** i
            # 计算新的参数参量,公式：new_para ≈ θ_old + α^i * H^-1 * g，是参数的更新尝试
            new_para = old_para + coef * max_vec
            # 创建新的策略网络副本，采用copy.deepcopy()避免修改原网络
            new_actor = copy.deepcopy(self.actor)
            # 将新的参数给副本网络
            torch.nn.utils.convert_parameters.vector_to_parameters(new_para, new_actor.parameters())
            # 计算新策略的动作的概率分布
            mu, std = new_actor(states)
            new_action_dicts = torch.distributions.Normal(mu, std)
            # 计算新策略与旧策略之间的KL散度
            kl_div = torch.mean(torch.distributions.kl.kl_divergence(old_action_dicts, new_action_dicts))
            # 计算新的策略的代理目标的值
            new_obj = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, new_actor)
            # 判断新的参数下是否满足约束
            if kl_div < self.KL_constraint and new_obj > old_obj:
                return new_para
        return old_para
    
    def policy_update(self, states, actions, old_action_dicts, old_log_probs, advantage):
        """
        更新策略网络参数，使用 TRPO 算法优化代理目标。
        该方法结合共轭梯度法和线性搜索，确保策略改进并满足 KL 散度约束。

        参数:
        - states (torch.Tensor): 状态张量，形状为 [batch_size, state_dim]
        - actions (torch.Tensor): 动作张量，形状为 [batch_size, 1]
        - old_action_dists (torch.distributions.Categorical): 旧策略动作分布
        - old_log_probs (torch.Tensor): 旧策略的对数概率，形状为 [batch_size, 1]
        - advantage (torch.Tensor): 优势函数值，形状为 [batch_size, 1]

        无返回值: 直接更新 self.actor 的参数
        """
        # 计算当前的代理目标值，用于后续的优化基准
        surrogate = self.compute_surrogate_obj(states, actions, advantage, old_log_probs, self.actor)
        # 计算代理目标函数的梯度
        grads = torch.autograd.grad(surrogate, self.actor.parameters())
        # 将梯度展平为一个向量，并分离计算图避免梯度积累,obj_grad = g，是展平后的梯度向量 g = ∇_θ L_θ_old(θ)|_θ=θ_old。detach() 防止梯度回溯，提高效率。
        surrogate_grad = torch.cat([grad.view(-1) for grad in grads]).detach()
        # 使用共轭梯度法计算自然梯度方向 x = H^-1 * g,descent_direction ≈ H^-1 * g，其中 H 是 Fisher 信息矩阵，
        # g 是梯度向量。共轭梯度法高效求解 Hx = g，x 是自然梯度方向。
        descent_direction = self.conjugate_gradient(surrogate_grad, states, old_action_dicts)
        # 计算 Hessian-向量乘积 H * descent_direction，Hd = H * (H^-1 * g)，验证共轭梯度法的中间结果。
        Hd = self.hessian_matrix_vector_product(states, old_action_dicts, descent_direction)
        # 计算最大步长系数，确保KL散度约束条件，max_coef = sqrt(2δ / (g^T H^-1 g))，δ 是 KL 约束阈值。
        max_coef = torch.sqrt(2 * self.KL_constraint / (torch.dot(descent_direction, Hd)+1e-8))
        # 执行线性搜索，找到满足步长的最大优化参数,new_para = θ_old + α * H^-1 * g，α 从 max_coef 逐步减小。
        new_para = self.line_search(states, actions, advantage, old_log_probs, old_action_dicts, descent_direction * max_coef)
        # 将优化好的参数加载到策略网络
        torch.nn.utils.convert_parameters.vector_to_parameters(new_para, self.actor.parameters())
    
    def update(self, transition_dict):
        states = torch.tensor(transition_dict['states'], dtype=torch.float).to(self.device)
        actions = torch.tensor(transition_dict['actions'], dtype=torch.float).view(-1,1).to(self.device)
        rewards = torch.tensor(transition_dict['rewards'], dtype=torch.float).view(-1,1).to(self.device)
        next_states = torch.tensor(transition_dict['next_states'], dtype=torch.float).to(self.device)
        dones = torch.tensor(transition_dict['dones'], dtype=torch.float).view(-1,1).to(self.device)
        # 倒立摆对应环境的原始奖励范围在[-16,0]，为了方便计算，将奖励范围映射到[-1,1]
        rewards = (rewards + 8) / 8
        td_target = rewards + self.gamma * self.critic(next_states) * (1-dones)
        td_error = td_target - self.critic(states)
        # 计算优势函数,td_error在GPU中，外部计算优势函数以numpy为核心，在cpu上
        advantage = compute_advantage(self.gamma, self.lmbda, td_error.cpu()).to(self.device)
        # 计算旧的动作分布
        mu, std = self.actor(states)
        old_action_dicts = torch.distributions.Normal(mu.detach(), std.detach())
        # 计算旧的对数策略分布概率
        old_log_probs = old_action_dicts.log_prob(actions)
        # 更新Critic网络参数
        critic_loss = torch.mean(F.mse_loss(self.critic(states), td_target.detach()))
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        # 更新策略网络
        self.policy_update(states, actions, old_action_dicts, old_log_probs, advantage)       

        