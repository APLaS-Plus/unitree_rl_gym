"""
PPO 训练脚本 (rsl_rl) - 适配 legged_gym 环境

使用 rsl_rl 的 PPO 算法进行训练，用于与自定义 SAC 算法对比实验
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
import time
import statistics
from collections import deque
from argparse import Namespace
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from rich.progress import track

from rsl_rl.algorithms import PPO
from models import PPOActorCritic  # 使用自定义 ActorCritic

# ============================================
# 配置参数
# ============================================
# 环境配置
task = "go2"
num_envs = 512
headless = False
sim_device = "cuda:0"
rl_device = "cuda:0"

# 通用超参数（与 SAC 保持一致）
gamma = 0.99
n_steps = 24                  # 每次更新前收集的步数 (per env)

# 训练迭代配置
# max_iterations: PPO 更新次数 (每次更新收集 n_steps * num_envs 个环境步)
# 总环境步数 = max_iterations * n_steps * num_envs
max_iterations = 15000        # PPO 更新次数 (约 15000 * 24 * 256 ≈ 92M 环境步)
log_freq = 10                 # 打印日志的频率 (每 N 次更新)
eval_freq = 100               # 评估频率 (每 N 次更新)
save_freq = 500               # 保存模型频率 (每 N 次更新)
hidden_dim = [256, 512, 512, 256]  # 网络隐藏层维度

# PPO 特有超参数
# ============================================
learning_rate_start = 3e-4    # 初始学习率
learning_rate_end = 5e-6      # 最终学习率（线性衰减）
num_mini_batches = 4          # mini-batch 数量
num_learning_epochs = 5       # 每次更新的 epoch 数
clip_param = 0.2              # PPO clip 范围
entropy_coef = 0.01           # 熵系数 (鼓励探索)
value_loss_coef = 0.5         # Value function loss 系数
max_grad_norm = 1.0           # 梯度裁剪
gae_lambda = 0.97             # GAE lambda
init_noise_std = 1.0          # 初始动作噪声标准差
# ============================================

# 路径设置
ROOT_PATH = Path(__file__).parent.resolve()
runs_base_path = ROOT_PATH / "runs_ppo"
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


def linear_schedule(initial_value: float, final_value: float, progress: float) -> float:
    """
    线性学习率调度器
    
    Args:
        initial_value: 初始学习率
        final_value: 最终学习率
        progress: 训练进度 (0.0 -> 1.0)
        
    Returns:
        当前学习率
    """
    return initial_value + progress * (final_value - initial_value)


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
        subscenes=4,
        num_threads=0,
        device="cuda",
        compute_device_id=0,
        sim_device_type="cuda",
        sim_device_id=0
    )
    
    env, env_cfg = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    
    obs_dim = env.num_obs
    action_dim = env.num_actions
    
    print(f"观测维度: {obs_dim}")
    print(f"动作维度: {action_dim}")
    print(f"环境数量: {env.num_envs}")
    
    # ============================================
    # 2. 创建 PPO 模型
    # ============================================
    print("\n创建 PPO 模型...")
    print(f"网络结构: {hidden_dim}")
    print(f"学习率调度: {learning_rate_start} -> {learning_rate_end} (线性衰减)")
    
    # 创建 Actor-Critic 网络
    actor_critic = PPOActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,  # 使用相同的观测
        num_actions=action_dim,
        actor_hidden_dims=hidden_dim,
        critic_hidden_dims=hidden_dim,
        activation='mish',
        init_noise_std=init_noise_std,
    )
    
    # 创建 PPO 算法
    ppo = PPO(
        actor_critic=actor_critic,
        num_learning_epochs=num_learning_epochs,
        num_mini_batches=num_mini_batches,
        clip_param=clip_param,
        gamma=gamma,
        lam=gae_lambda,
        value_loss_coef=value_loss_coef,
        entropy_coef=entropy_coef,
        learning_rate=learning_rate_start,
        max_grad_norm=max_grad_norm,
        use_clipped_value_loss=True,
        schedule="fixed",  # 我们自己实现学习率调度
        device=rl_device,
    )
    
    # 初始化 storage
    ppo.init_storage(
        num_envs=num_envs,
        num_transitions_per_env=n_steps,
        actor_obs_shape=[obs_dim],
        critic_obs_shape=[obs_dim],
        action_shape=[action_dim],
    )
    
    # ============================================
    # 3. TensorBoard
    # ============================================
    writer = SummaryWriter(log_dir=str(tensorboard_path))
    
    # ============================================
    # 4. 训练循环
    # ============================================
    print(f"\n开始训练, 最大迭代: {max_iterations}")
    
    obs = env.get_observations()
    env._resample_commands(torch.arange(num_envs, device=rl_device))
    obs = env.get_observations()
    
    # 统计变量 - 使用 deque 保持最近 100 个 episode 的统计
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=rl_device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=rl_device)
    
    # Episode info 统计 (用于记录环境返回的奖励分量等信息)
    ep_infos = []
    
    total_steps = 0
    tot_time = 0.0
    num_updates = max_iterations  # max_iterations 直接表示 PPO 更新次数
    
    print(f"总更新次数: {num_updates}")
    
    ppo.train_mode()
    
    for update_idx in track(range(num_updates), description="训练中"):
        start_time = time.time()
        
        # 更新学习率 (线性衰减)
        progress = update_idx / num_updates
        current_lr = linear_schedule(learning_rate_start, learning_rate_end, progress)
        for param_group in ppo.optimizer.param_groups:
            param_group['lr'] = current_lr
        
        # ============================================
        # 4.1 收集数据 (Rollout)
        # ============================================
        with torch.inference_mode():
            for step in range(n_steps):
                # 选择动作
                actions = ppo.act(obs, obs)
                
                # 执行动作
                next_obs, privileged_obs, rewards, dones, infos = env.step(actions)
                
                # 处理环境步骤
                ppo.process_env_step(rewards, dones, infos)
                
                # Episode info 收集 (奖励分量等)
                if 'episode' in infos:
                    ep_infos.append(infos['episode'])
                
                # 统计完成的 episode
                cur_reward_sum += rewards
                cur_episode_length += 1
                total_steps += num_envs
                
                # 处理完成的 episode
                new_ids = (dones > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                    lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0
                
                obs = next_obs
        
        collection_time = time.time() - start_time
        
        # ============================================
        # 4.2 计算 returns 并更新策略
        # ============================================
        learn_start = time.time()
        
        with torch.no_grad():
            ppo.compute_returns(obs)
        
        # 在更新前获取 storage 中的统计信息
        advantages = ppo.storage.advantages
        returns = ppo.storage.returns
        values = ppo.storage.values
        
        # Advantage 统计
        adv_mean = advantages.mean().item()
        adv_std = advantages.std().item()
        adv_min = advantages.min().item()
        adv_max = advantages.max().item()
        
        # Value 统计
        value_mean = values.mean().item()
        value_std = values.std().item()
        
        # Returns 统计
        returns_mean = returns.mean().item()
        returns_std = returns.std().item()
        
        # 计算 Explained Variance: 衡量 value function 对 returns 的解释程度
        # 接近 1 表示 value function 很好地预测了 returns
        # 接近 0 或负数表示 value function 预测很差
        y_pred = values.flatten().cpu().numpy()
        y_true = returns.flatten().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0
        
        # 获取更新前的 log_prob 用于计算 KL 散度
        old_actions_log_prob = ppo.storage.actions_log_prob.clone()
        old_mu = ppo.storage.mu.clone()
        old_sigma = ppo.storage.sigma.clone()
        
        # PPO 更新
        mean_value_loss, mean_surrogate_loss = ppo.update()
        
        learn_time = time.time() - learn_start
        tot_time += collection_time + learn_time
        
        # ============================================
        # 4.3 计算额外的训练指标
        # ============================================
        
        # 策略噪声标准差 (探索程度)
        mean_std = ppo.actor_critic.std.mean().item()
        
        # 计算 KL 散度 (新旧策略之间的差异)
        # 使用高斯分布的 KL 散度公式
        with torch.no_grad():
            # 获取当前策略的参数
            _ = ppo.actor_critic.act(obs)
            new_mu = ppo.actor_critic.action_mean
            new_sigma = ppo.actor_critic.action_std
            
            # KL(old || new) for Gaussian
            # 注意: old_mu/old_sigma 是整个 rollout 的平均，这里用最后一个 step 近似
            kl = torch.sum(
                torch.log(new_sigma / (old_sigma.mean(dim=0) + 1e-5) + 1e-5) + 
                (old_sigma.mean(dim=0).pow(2) + (old_mu.mean(dim=0) - new_mu).pow(2)) / 
                (2.0 * new_sigma.pow(2) + 1e-5) - 0.5, 
                dim=-1
            )
            kl_mean = kl.mean().item()
        
        # 计算熵 (探索程度的另一个指标)
        entropy = ppo.actor_critic.entropy.mean().item() if hasattr(ppo.actor_critic, 'entropy') else 0.0
        
        # ============================================
        # 4.4 记录到 TensorBoard
        # ============================================
        it = update_idx  # iteration number
        
        # Loss
        writer.add_scalar("Loss/value_function", mean_value_loss, it)
        writer.add_scalar("Loss/surrogate", mean_surrogate_loss, it)
        
        # Learning Rate
        writer.add_scalar("Loss/learning_rate", current_lr, it)
        
        # Policy / Exploration
        writer.add_scalar("Policy/mean_noise_std", mean_std, it)
        writer.add_scalar("Policy/entropy", entropy, it)
        writer.add_scalar("Policy/kl_divergence", kl_mean, it)
        
        # Value Function
        writer.add_scalar("Value/mean", value_mean, it)
        writer.add_scalar("Value/std", value_std, it)
        writer.add_scalar("Value/explained_variance", explained_var, it)
        
        # Advantage
        writer.add_scalar("Advantage/mean", adv_mean, it)
        writer.add_scalar("Advantage/std", adv_std, it)
        writer.add_scalar("Advantage/min", adv_min, it)
        writer.add_scalar("Advantage/max", adv_max, it)
        
        # Returns
        writer.add_scalar("Returns/mean", returns_mean, it)
        writer.add_scalar("Returns/std", returns_std, it)
        
        # Performance
        fps = int(n_steps * num_envs / (collection_time + learn_time))
        writer.add_scalar("Perf/total_fps", fps, it)
        writer.add_scalar("Perf/collection_time", collection_time, it)
        writer.add_scalar("Perf/learning_time", learn_time, it)
        
        # Episode Info (奖励分量等)
        if ep_infos:
            for key in ep_infos[0]:
                infotensor = torch.tensor([], device=rl_device)
                for ep_info in ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(rl_device)))
                value = torch.mean(infotensor)
                writer.add_scalar('Episode/' + key, value, it)
        
        # Train metrics (使用 buffer 中的统计)
        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)
            mean_ep_len = statistics.mean(lenbuffer)
            std_reward = statistics.stdev(rewbuffer) if len(rewbuffer) > 1 else 0
            std_ep_len = statistics.stdev(lenbuffer) if len(lenbuffer) > 1 else 0
            
            writer.add_scalar('Train/mean_reward', mean_reward, it)
            writer.add_scalar('Train/std_reward', std_reward, it)
            writer.add_scalar('Train/mean_episode_length', mean_ep_len, it)
            writer.add_scalar('Train/std_episode_length', std_ep_len, it)
            writer.add_scalar('Train/mean_reward/time', mean_reward, tot_time)
            writer.add_scalar('Train/mean_episode_length/time', mean_ep_len, tot_time)
        
        # ============================================
        # 4.5 定期打印统计信息 (格式化输出)
        # ============================================
        if update_idx % log_freq == 0:
            width = 80
            pad = 35
            
            log_string = f"\n{'#' * width}\n"
            log_string += f" Learning iteration {update_idx}/{num_updates} ".center(width, ' ') + "\n\n"
            log_string += f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning: {learn_time:.3f}s)\n"
            log_string += f"{'Value function loss:':>{pad}} {mean_value_loss:.4f}\n"
            log_string += f"{'Surrogate loss:':>{pad}} {mean_surrogate_loss:.4f}\n"
            log_string += f"{'Mean action noise std:':>{pad}} {mean_std:.4f}\n"
            log_string += f"{'Entropy:':>{pad}} {entropy:.4f}\n"
            log_string += f"{'KL Divergence:':>{pad}} {kl_mean:.6f}\n"
            log_string += f"{'Learning rate:':>{pad}} {current_lr:.2e}\n"
            log_string += f"{'Explained Variance:':>{pad}} {explained_var:.4f}\n"
            log_string += f"{'Value (mean/std):':>{pad}} {value_mean:.2f} / {value_std:.2f}\n"
            log_string += f"{'Advantage (mean/std):':>{pad}} {adv_mean:.4f} / {adv_std:.4f}\n"
            log_string += f"{'Returns (mean/std):':>{pad}} {returns_mean:.2f} / {returns_std:.2f}\n"
            
            if len(rewbuffer) > 0:
                log_string += f"{'Mean reward:':>{pad}} {mean_reward:.2f} (std: {std_reward:.2f})\n"
                log_string += f"{'Mean episode length:':>{pad}} {mean_ep_len:.2f} (std: {std_ep_len:.2f})\n"
            
            # Episode info 详细信息
            if ep_infos:
                for key in ep_infos[0]:
                    infotensor = torch.tensor([], device=rl_device)
                    for ep_info in ep_infos:
                        if not isinstance(ep_info[key], torch.Tensor):
                            ep_info[key] = torch.Tensor([ep_info[key]])
                        if len(ep_info[key].shape) == 0:
                            ep_info[key] = ep_info[key].unsqueeze(0)
                        infotensor = torch.cat((infotensor, ep_info[key].to(rl_device)))
                    value = torch.mean(infotensor)
                    log_string += f"{f'Episode/{key}:':>{pad}} {value:.4f}\n"
            
            log_string += f"{'-' * width}\n"
            log_string += f"{'Total timesteps:':>{pad}} {total_steps}\n"
            log_string += f"{'Total time:':>{pad}} {tot_time:.2f}s\n"
            eta = tot_time / (update_idx + 1) * (num_updates - update_idx) if update_idx > 0 else 0
            log_string += f"{'ETA:':>{pad}} {eta:.1f}s\n"
            
            print(log_string)
        
        # 清空 episode info buffer
        ep_infos.clear()
        
        # ============================================
        # 4.6 保存模型
        # ============================================
        if update_idx > 0 and update_idx % save_freq == 0:
            checkpoint_path = model_save_path / "checkpoints"
            os.makedirs(str(checkpoint_path), exist_ok=True)
            torch.save({
                'update_idx': update_idx,
                'total_steps': total_steps,
                'actor_critic': actor_critic.state_dict(),
                'optimizer': ppo.optimizer.state_dict(),
            }, checkpoint_path / f"checkpoint_{update_idx}.pt")
            print(f"模型已保存: {checkpoint_path}/checkpoint_{update_idx}.pt")
    
    # 保存最终模型
    torch.save({
        'actor_critic': actor_critic.state_dict(),
        'optimizer': ppo.optimizer.state_dict(),
    }, model_save_path / "final_model.pt")
    
    writer.close()
    print(f"\n训练完成! 模型保存在: {model_save_path}")


if __name__ == '__main__':
    main()
