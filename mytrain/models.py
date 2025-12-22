import os
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    """
    参数:
        layer: 网络层
        std: 正交初始化的增益 (gain)
        bias_const: 偏置初始化的常数值
    """
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
    return layer

class RuningMeanStd(nn.Module):
    def __init__(self, shape, epsilon=1e-5):
        super(RuningMeanStd, self).__init__()
        self.shape = shape
        self.epsilon = epsilon

        self.register_buffer('count', torch.tensor(data=epsilon, dtype=torch.float64))
        self.register_buffer('mean', torch.zeros(size=(shape,), dtype=torch.float64))
        self.register_buffer('var', torch.ones(size=(shape,), dtype=torch.float64))

    def update(self, x):
        batch_mean = torch.mean(x, dim=0)
        batch_var = torch.var(x, dim=0, unbiased=False)
        batch_count = x.size(0)

        delta = batch_mean - self.mean
        total_count = self.count + batch_count

        self.mean = self.mean + delta * batch_count / total_count

        m2 = self.var * self.count + batch_var * batch_count + torch.square(delta) * self.count * batch_count / total_count

        self.var = m2 / total_count
        self.count = total_count

    def forward(self, x, should_update = False):
        x = torch.tensor(data=x, dtype=torch.float64)
        if should_update:
            self.update(x)

        result = (x - self.mean.to(x.device).to(x.dtype)) / torch.sqrt(self.var.to(x.device).to(x.dtype) + self.epsilon)
        return result.to(torch.float32)

    def save_state_dict(self, path: Path):
        os.makedirs(str(path.parent), exist_ok=True)
        torch.save(self.state_dict(), path)
        

class DQNAgent(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = (128, 128, 128)):
        super(DQNAgent, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        layers = [nn.Linear(obs_dim, self.hidden_dim[0])]
        for i in range(len(self.hidden_dim) - 1):
            # layers.append(nn.LayerNorm(self.hidden_dim[i]))
            layers.append(nn.Tanh())
            if i != len(self.hidden_dim):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        layers.append(nn.Linear(hidden_dim[-1], action_dim))
        
        self.func = nn.Sequential(*layers)
        
        # self.init_parm()

    def forward(self, obs):
        # print(obs.size(-1))
        assert obs.size(-1) == self.obs_dim
        return self.func(obs)

    # def init_parm(self):
    #     for layer in self.func:
    #         if isinstance(layer, (nn.Linear, nn.LayerNorm)):
    #             nn.init.orthogonal_(layer.weight, gain=nn.init.calculate_gain('tanh'))
    #             nn.init.constant_(layer.bias, 0.0)

    def get_action(self, obs):
        with torch.no_grad():
            q = self.forward(torch.Tensor(obs).to(device).to(torch.float32).unsqueeze(0))
            return torch.argmax(q).item()

    def save_state_dict(self, path: Path):
        os.makedirs(str(path.parent), exist_ok=True)
        torch.save(self.state_dict(), path)


class SimpleActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_high, hidden_dim = (128, 128, 128)):
        super(SimpleActor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_high = action_high

        layers = [nn.Linear(obs_dim, self.hidden_dim[0])]
        for i in range(len(self.hidden_dim) - 1):
            # layers.append(nn.LayerNorm(self.hidden_dim[i]))
            layers.append(nn.Tanh())
            if i != len(self.hidden_dim):
                layers.append(nn.Linear(hidden_dim[i], hidden_dim[i+1]))
        layers.append(nn.Linear(hidden_dim[-1], action_dim))
        layers.append(nn.Tanh())
        
        self.func = nn.Sequential(*layers)

    def forward(self, obs):
        # print(obs.size(-1))
        assert obs.size(-1) == self.obs_dim
        return self.func(obs) * self.action_high

    def get_action(self, obs):
        with torch.no_grad():
            action = self.forward(torch.Tensor(obs).to(device).to(torch.float32).unsqueeze(0))
            return action.cpu().numpy().flatten()

    def save_state_dict(self, path: Path):
        os.makedirs(str(path.parent), exist_ok=True)
        torch.save(self.state_dict(), path)

class StochasticActor(nn.Module):
    def __init__(self, obs_dim, action_dim, action_high, hidden_dim = (128, 128, 128), epsilon=1e-6):
        super(StochasticActor, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.action_high = action_high
        self.epsilon = epsilon
        
        layers = [layer_init(nn.Linear(obs_dim, self.hidden_dim[0])), nn.LayerNorm(self.hidden_dim[0])]
        for i in range(len(self.hidden_dim) - 1):
            layers.append(layer_init(nn.Mish()))
            if i != len(self.hidden_dim):
                layers.append(layer_init(nn.Linear(hidden_dim[i], hidden_dim[i+1])))
                layers.append(nn.LayerNorm(self.hidden_dim[i+1]))

        self.trunk = nn.Sequential(*layers)
        
        self.fn_mean = nn.Sequential(layer_init(nn.Linear(hidden_dim[-1], action_dim), std=1.0), nn.LayerNorm(self.action_dim))
        self.fn_logstd = nn.Sequential(layer_init(nn.Linear(hidden_dim[-1], action_dim), std=1.0), nn.LayerNorm(self.action_dim))

    def forward(self, obs):
        assert obs.size(-1) == self.obs_dim
        x = self.trunk(obs)
        mean = self.fn_mean(x)
        logstd = self.fn_logstd(x)
        logstd = torch.clamp(logstd, -20, 2)

        return mean, logstd

    def sample(self, obs):
        mean, logstd = self.forward(obs)
        std = torch.exp(logstd)

        dist = torch.distributions.Normal(mean, std)

        x = dist.rsample()

        y = torch.tanh(x)

        action = self.action_high*y

        logprob = dist.log_prob(x)
        logprob = logprob - torch.log(self.action_high * (1 - y.pow(2)) + self.epsilon)
        logprob = logprob.sum(dim=-1, keepdim=True)

        return action, logprob

    def get_action(self, obs):
        with torch.no_grad():
            obs_tensor = torch.Tensor(obs).to(device).to(torch.float32).unsqueeze(0)
            mean, _ = self.forward(obs_tensor)
            action = torch.tanh(mean) * self.action_high 
            return action.cpu().numpy().flatten()

    def save_state_dict(self, path: Path):
        os.makedirs(str(path.parent), exist_ok=True)
        torch.save(self.state_dict(), path)


class Critic(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim = (128, 128, 128)):
        super(Critic, self).__init__()

        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        layers = [layer_init(nn.Linear(obs_dim + action_dim, self.hidden_dim[0]))]
        for i in range(len(self.hidden_dim) - 1):
            # layers.append(nn.LayerNorm(self.hidden_dim[i]))
            layers.append(layer_init(nn.Mish()))
            if i != len(self.hidden_dim):
                layers.append(layer_init(nn.Linear(hidden_dim[i], hidden_dim[i+1])))
        layers.append(layer_init(nn.Linear(hidden_dim[-1], 1), std=1))
        
        self.func = nn.Sequential(*layers)

    def forward(self, obs, action):
        # print(obs.size(-1))
        assert obs.size(-1) == self.obs_dim
        assert action.size(-1) == self.action_dim
        x = torch.cat([obs, action], dim=-1)
        return self.func(x)

    def save_state_dict(self, path: Path):
        os.makedirs(str(path.parent), exist_ok=True)
        torch.save(self.state_dict(), path)


# ============================================
# PPO Actor-Critic (兼容 rsl_rl.PPO)
# ============================================

def get_activation(act_name):
    """获取激活函数"""
    activations = {
        "elu": nn.ELU(),
        "selu": nn.SELU(),
        "relu": nn.ReLU(),
        "mish": nn.Mish(),
        "lrelu": nn.LeakyReLU(),
        "tanh": nn.Tanh(),
        "sigmoid": nn.Sigmoid(),
        "gelu": nn.GELU(),
        "mish": nn.Mish(),
    }
    if act_name not in activations:
        raise ValueError(f"Unknown activation: {act_name}. Available: {list(activations.keys())}")
    return activations[act_name]


class PPOActorCritic(nn.Module):
    """
    自定义 PPO Actor-Critic 网络，兼容 rsl_rl.PPO 算法
    
    特性:
    - 支持多种激活函数 (elu, mish, relu, gelu, etc.)
    - 可选的 LayerNorm
    - 正交初始化权重
    - 与 rsl_rl.PPO 完全兼容的接口
    
    Usage:
        actor_critic = PPOActorCritic(
            num_actor_obs=48,
            num_critic_obs=48,
            num_actions=12,
            actor_hidden_dims=[512, 512, 256],
            critic_hidden_dims=[512, 512, 256],
            activation='mish',
            use_layer_norm=False,
            init_noise_std=1.0,
        )
    """
    is_recurrent = False  # rsl_rl 需要这个属性
    
    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        actor_hidden_dims: list = [256, 256, 256],
        critic_hidden_dims: list = [256, 256, 256],
        activation: str = 'elu',
        init_noise_std: float = 1.0,
        use_layer_norm: bool = False,
        ortho_init: bool = True,
        **kwargs
    ):
        if kwargs:
            print(f"PPOActorCritic: 忽略未知参数: {list(kwargs.keys())}")
        super(PPOActorCritic, self).__init__()
        
        self.num_actor_obs = num_actor_obs
        self.num_critic_obs = num_critic_obs
        self.num_actions = num_actions
        
        # 激活函数
        act_fn = get_activation(activation)
        
        # ============ Actor 网络 ============
        actor_layers = []
        actor_layers.append(nn.Linear(num_actor_obs, actor_hidden_dims[0]))
        if use_layer_norm:
            actor_layers.append(nn.LayerNorm(actor_hidden_dims[0]))
        actor_layers.append(act_fn)
        
        for i in range(len(actor_hidden_dims) - 1):
            actor_layers.append(nn.Linear(actor_hidden_dims[i], actor_hidden_dims[i + 1]))
            if use_layer_norm:
                actor_layers.append(nn.LayerNorm(actor_hidden_dims[i + 1]))
            actor_layers.append(get_activation(activation))
        
        actor_layers.append(nn.Linear(actor_hidden_dims[-1], num_actions))
        self.actor = nn.Sequential(*actor_layers)
        
        # ============ Critic 网络 ============
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_obs, critic_hidden_dims[0]))
        if use_layer_norm:
            critic_layers.append(nn.LayerNorm(critic_hidden_dims[0]))
        critic_layers.append(get_activation(activation))
        
        for i in range(len(critic_hidden_dims) - 1):
            critic_layers.append(nn.Linear(critic_hidden_dims[i], critic_hidden_dims[i + 1]))
            if use_layer_norm:
                critic_layers.append(nn.LayerNorm(critic_hidden_dims[i + 1]))
            critic_layers.append(get_activation(activation))
        
        critic_layers.append(nn.Linear(critic_hidden_dims[-1], 1))
        self.critic = nn.Sequential(*critic_layers)
        
        # ============ 动作噪声 (可学习的标准差) ============
        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution = None
        
        # 禁用分布验证以加速
        torch.distributions.Normal.set_default_validate_args = False
        
        # 正交初始化
        if ortho_init:
            self._init_weights()
        
        print(f"PPOActorCritic 网络结构:")
        print(f"  Actor: {self.actor}")
        print(f"  Critic: {self.critic}")
        print(f"  初始噪声标准差: {init_noise_std}")
    
    def _init_weights(self):
        """正交初始化权重"""
        for module in self.actor.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        for module in self.critic.modules():
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                nn.init.constant_(module.bias, 0.0)
        
        # 最后一层使用更小的增益
        if isinstance(self.actor[-1], nn.Linear):
            nn.init.orthogonal_(self.actor[-1].weight, gain=0.01)
        if isinstance(self.critic[-1], nn.Linear):
            nn.init.orthogonal_(self.critic[-1].weight, gain=1.0)
    
    def reset(self, dones=None):
        """重置隐藏状态 (非循环网络不需要)"""
        pass
    
    def forward(self):
        raise NotImplementedError("Use act() or evaluate() instead")
    
    # ============ 分布相关属性 (rsl_rl 需要) ============
    @property
    def action_mean(self):
        return self.distribution.mean
    
    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)
    
    # ============ 核心方法 ============
    def update_distribution(self, observations):
        """更新动作分布"""
        mean = self.actor(observations)
        self.distribution = torch.distributions.Normal(mean, self.std)
    
    def act(self, observations, **kwargs):
        """采样动作 (训练时使用)"""
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        """计算动作的 log 概率"""
        return self.distribution.log_prob(actions).sum(dim=-1)
    
    def act_inference(self, observations):
        """推理时使用 (确定性动作)"""
        return self.actor(observations)
    
    def evaluate(self, critic_observations, **kwargs):
        """评估状态价值"""
        return self.critic(critic_observations)
    
    # ============ 保存/加载 ============
    def save_state_dict(self, path: Path):
        os.makedirs(str(path.parent), exist_ok=True)
        torch.save(self.state_dict(), path)