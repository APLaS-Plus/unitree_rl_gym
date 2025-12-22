"""
SAC 训练脚本 - 适配 legged_gym 环境
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import task_registry

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import LazyTensorStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from argparse import Namespace
from rich.progress import track

from models import StochasticActor, Critic, device

# ============================================
# 配置参数
# ============================================

torch.backends.cuda.matmul.allow_tf32 = True

# 环境配置
task = "go2"
num_envs = 1024
headless = True
sim_device = "cuda:0"
rl_device = "cuda:0"

# SAC 超参数
gamma = 0.99
tau = 0.005
actor_lr = 1e-5
critic_lr = 1e-4
alpha_lr = 1e-6
batch_size = 256
rb_size = 3_000_000
start_steps = 1000  # 随机探索步数
max_iterations = 2_000_000
eval_freq = 5000
save_freq = 10000
train_times_per_collect = 4
hidden_dim = (512, 512, 512)

# 路径设置
ROOT_PATH = Path(__file__).parent.resolve()
runs_base_path = ROOT_PATH / "runs_sac"
os.makedirs(str(runs_base_path), exist_ok=True)

run_num = 1
while (runs_base_path / f"run{run_num}").exists():
    run_num += 1

model_save_path = runs_base_path / f"run{run_num}"
tensorboard_path = runs_base_path / "tensorboard" / f"run{run_num}"
os.makedirs(str(model_save_path), exist_ok=True)
os.makedirs(str(tensorboard_path), exist_ok=True)

print(f"保存路径: {model_save_path}")
print(f"TensorBoard: {tensorboard_path}")


def soft_update(target, online, tau):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)


def main():
    # ============================================
    # 1. 创建环境
    # ============================================
    print(f"\n创建环境: task={task}, num_envs={num_envs}")
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_envs
    
    args = Namespace(
        task=task,
        num_envs=num_envs,
        headless=headless,
        sim_device=sim_device,
        rl_device=rl_device,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=True,
        use_gpu_pipeline=True,
        subscenes=0,
        num_threads=0,
        device="cuda",
        compute_device_id=0,
        sim_device_type="cuda",
        sim_device_id=0
    )
    
    env, env_cfg = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    
    obs_dim = env.num_obs
    action_dim = env.num_actions
    action_high = torch.ones(action_dim, device=device)  # 动作范围 [-1, 1]
    
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"环境数量: {env.num_envs}")
    
    # ============================================
    # 2. 创建模型
    # ============================================
    print("\n创建模型...")
    
    actor = StochasticActor(obs_dim, action_dim, action_high, hidden_dim).to(device)
    critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    
    target_critic1 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    target_critic2 = Critic(obs_dim, action_dim, hidden_dim).to(device)
    target_critic1.load_state_dict(critic1.state_dict())
    target_critic2.load_state_dict(critic2.state_dict())
    target_critic1.eval()
    target_critic2.eval()
    
    # 自动温度参数
    log_alpha = torch.zeros(1, requires_grad=True, device=device)
    target_entropy = -action_dim  # 目标熵
    
    # 优化器
    actor_optim = torch.optim.Adam(actor.parameters(), lr=actor_lr)
    critic_optim = torch.optim.Adam(
        list(critic1.parameters()) + list(critic2.parameters()), lr=critic_lr
    )
    alpha_optim = torch.optim.Adam([log_alpha], lr=alpha_lr)
    
    loss_fn = nn.MSELoss(reduction="mean")
    
    # ============================================
    # 3. Replay Buffer
    # ============================================
    print(f"创建 Replay Buffer, 容量: {rb_size}")
    
    rb = TensorDictReplayBuffer(
        storage=LazyTensorStorage(max_size=rb_size, device=device),
        sampler=RandomSampler(),
        batch_size=batch_size,
    )
    
    # ============================================
    # 4. TensorBoard
    # ============================================
    writer = SummaryWriter(log_dir=str(tensorboard_path))
    
    # ============================================
    # 5. 训练循环
    # ============================================
    print(f"\n开始训练, 最大迭代: {max_iterations}")
    print(f"随机探索步数: {start_steps}")
    
    obs = env.get_observations()
    
    # 初始化速度指令（让机器人开始行走）
    env._resample_commands(torch.arange(num_envs, device=device))
    obs = env.get_observations()  # 重新获取包含指令的观测
    episode_rewards = torch.zeros(num_envs, device=device)
    episode_lengths = torch.zeros(num_envs, device=device)
    completed_episodes = 0
    total_reward_sum = 0.0
    
    for step in track(range(max_iterations), description="训练中"):
        # 选择动作
        if step < start_steps:
            # 随机探索
            actions = torch.rand(num_envs, action_dim, device=device) * 2 - 1
        else:
            with torch.no_grad():
                actions, _ = actor.sample(obs)
        
        # 执行动作
        next_obs, privileged_obs, rewards, dones, infos = env.step(actions)
        rewards = (rewards * 10.0).clamp(-10.0, 10.0)
        
        # 区分超时和真实终止（摔倒等）
        # 超时不应被视为 "done"，因为机器人还活着，下一时刻有价值
        real_dones = dones.clone()
        if "time_outs" in infos:
            real_dones = dones & ~infos["time_outs"]
        
        # 存储到 replay buffer
        data = TensorDict({
            "obs": obs,
            "action": actions,
            ("next", "obs"): next_obs,
            ("next", "reward"): rewards.unsqueeze(-1),
            ("next", "done"): real_dones.unsqueeze(-1).float(),  # 使用 real_dones
        }, batch_size=[num_envs])
        
        rb.extend(data)
        
        # 统计
        episode_rewards += rewards
        episode_lengths += 1
        
        # 处理完成的 episode
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            completed_episodes += len(done_ids)
            total_reward_sum += episode_rewards[done_ids].sum().item()
            episode_rewards[done_ids] = 0
            episode_lengths[done_ids] = 0
        
        obs = next_obs
        
        # 训练（在收集足够数据后）- 每收集一步训练多次
        if step >= start_steps and len(rb) >= batch_size:
            for train_idx in range(train_times_per_collect):
                sample = rb.sample()
                
                obs_batch = sample["obs"]
                action_batch = sample["action"]
                next_obs_batch = sample[("next", "obs")]
                reward_batch = sample[("next", "reward")]
                done_batch = sample[("next", "done")]
                
                # 计算 target Q
                with torch.no_grad():
                    next_actions, next_log_prob = actor.sample(next_obs_batch)
                    next_q1 = target_critic1(next_obs_batch, next_actions)
                    next_q2 = target_critic2(next_obs_batch, next_actions)
                    next_q = torch.min(next_q1, next_q2)
                    
                    alpha = log_alpha.exp()
                    next_q_value = next_q - alpha * next_log_prob
                    target_q = reward_batch + gamma * next_q_value * (1 - done_batch)
                
                # 更新 Critic
                curr_q1 = critic1(obs_batch, action_batch)
                curr_q2 = critic2(obs_batch, action_batch)
                critic1_loss = loss_fn(curr_q1, target_q)
                critic2_loss = loss_fn(curr_q2, target_q)
                critic_loss = critic1_loss + critic2_loss
                
                critic_optim.zero_grad()
                critic_loss.backward()
                critic_optim.step()
                
                # 更新 Actor
                actions_pi, log_prob_pi = actor.sample(obs_batch)
                q1_pi = critic1(obs_batch, actions_pi)
                q2_pi = critic2(obs_batch, actions_pi)
                min_q_pi = torch.min(q1_pi, q2_pi)
                
                alpha = log_alpha.exp().detach()
                actor_loss = (alpha * log_prob_pi - min_q_pi).mean()
                
                actor_optim.zero_grad()
                actor_loss.backward()
                actor_optim.step()
                
                # 更新 Alpha
                alpha_loss = -(log_alpha.exp() * (log_prob_pi + target_entropy).detach()).mean()
                
                alpha_optim.zero_grad()
                alpha_loss.backward()
                alpha_optim.step()
                
                # 软更新 target networks
                soft_update(target_critic1, critic1, tau)
                soft_update(target_critic2, critic2, tau)
            
            # 记录到 TensorBoard (每个 collect step 只记录一次，使用最后一次训练的数据)
            if step % 100 == 0:
                # === Loss ===
                writer.add_scalar("Loss/critic1", critic1_loss.item(), step)
                writer.add_scalar("Loss/critic2", critic2_loss.item(), step)
                writer.add_scalar("Loss/actor", actor_loss.item(), step)
                writer.add_scalar("Loss/alpha", alpha_loss.item(), step)
                
                # === Param ===
                writer.add_scalar("Param/alpha", log_alpha.exp().item(), step)
                writer.add_scalar("Param/log_alpha", log_alpha.item(), step)
                
                # === Entropy - 用于与 alpha 对比分析 ===
                mean_entropy = -log_prob_pi.mean().item()  # 策略的平均熵 (负的log概率均值)
                writer.add_scalar("Entropy/policy_entropy", mean_entropy, step)
                writer.add_scalar("Entropy/target_entropy", target_entropy, step)
                
                # === Q值统计 - 诊断值函数估计 ===
                writer.add_scalar("Q/q1_mean", curr_q1.mean().item(), step)
                writer.add_scalar("Q/q2_mean", curr_q2.mean().item(), step)
                writer.add_scalar("Q/target_q_mean", target_q.mean().item(), step)
                writer.add_scalar("Q/q_diff", (curr_q1 - curr_q2).abs().mean().item(), step)
                
                # === 动作统计 - 监控策略行为 ===
                writer.add_scalar("Action/mean", actions_pi.mean().item(), step)
                writer.add_scalar("Action/std", actions_pi.std().item(), step)
                writer.add_scalar("Action/abs_mean", actions_pi.abs().mean().item(), step)
                
                # === 梯度范数 - 检测梯度爆炸/消失 ===
                actor_grad_norm = sum(p.grad.norm().item() ** 2 for p in actor.parameters() if p.grad is not None) ** 0.5
                critic_grad_norm = sum(p.grad.norm().item() ** 2 for p in list(critic1.parameters()) + list(critic2.parameters()) if p.grad is not None) ** 0.5
                writer.add_scalar("Gradient/actor_grad_norm", actor_grad_norm, step)
                writer.add_scalar("Gradient/critic_grad_norm", critic_grad_norm, step)
                
                # === Buffer & Reward ===
                writer.add_scalar("Buffer/size", len(rb), step)
                writer.add_scalar("Reward/batch_mean", reward_batch.mean().item(), step)
        
        # 定期记录奖励
        if step % 1000 == 0 and completed_episodes > 0:
            mean_reward = total_reward_sum / completed_episodes
            writer.add_scalar("Train/mean_episode_reward", mean_reward, step)
            writer.add_scalar("Train/completed_episodes", completed_episodes, step)
            print(f"Step {step} | Episodes: {completed_episodes} | Mean Reward: {mean_reward:.2f}")
            completed_episodes = 0
            total_reward_sum = 0.0
        
        # 保存模型
        if step > 0 and step % save_freq == 0:
            save_path = model_save_path / "checkpoints"
            os.makedirs(str(save_path), exist_ok=True)
            torch.save({
                'step': step,
                'actor': actor.state_dict(),
                'critic1': critic1.state_dict(),
                'critic2': critic2.state_dict(),
                'log_alpha': log_alpha,
            }, save_path / f"checkpoint_{step}.pt")
            print(f"模型已保存: {save_path}/checkpoint_{step}.pt")
    
    # 保存最终模型
    torch.save({
        'actor': actor.state_dict(),
        'critic1': critic1.state_dict(),
        'critic2': critic2.state_dict(),
        'log_alpha': log_alpha,
    }, model_save_path / "final_model.pt")
    
    writer.close()
    print(f"\n训练完成! 模型保存在: {model_save_path}")


if __name__ == '__main__':
    main()
