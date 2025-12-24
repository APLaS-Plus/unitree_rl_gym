"""
PPG 训练脚本 (GAE 版本) - 适配 legged_gym 环境

使用自定义 GAE-PPG 算法进行训练，用于与 SAC 算法对比实验
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

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
from rich.progress import track

from models import PPGActorCritic, device, RolloutBuffer

# ============================================
# 配置参数
# ============================================
torch.backends.cuda.matmul.allow_tf32 = True

# 环境配置
task = "go2"
num_envs = 512
headless = False
sim_device = "cuda:0"
rl_device = "cuda:0"

# 通用超参数（与 SAC 保持一致）
gamma = 0.99
n_steps = 24  # 每次更新前收集的步数 (per env)

# 训练迭代配置
max_iterations = 30000  # PPO 更新次数
log_freq = 10  # 打印日志的频率 (每 N 次更新)
eval_freq = 100  # 评估频率 (每 N 次更新)
save_freq = 1000  # 保存模型频率 (每 N 次更新)
hidden_dim = [512, 512, 512]  # 网络隐藏层维度

# PPO 特有超参数
learning_rate_start = 3e-4  # 初始学习率
learning_rate_end = 5e-6  # 最终学习率（线性衰减）
num_mini_batches = 4  # mini-batch 数量
num_learning_epochs = 5  # 每次更新的 epoch 数
clip_param = 0.2  # PPO clip 范围
entropy_coef = 0.01  # 熵系数 (鼓励探索)
value_loss_coef = 0.5  # Value function loss 系数
max_grad_norm = 1.0  # 梯度裁剪
gae_lambda = 0.97  # GAE lambda
init_noise_std = 1.0  # 初始动作噪声标准差

# PPG
aux_update_freq = 32
n_aux_epochs = 6
beta_clone = 1

# 路径设置
ROOT_PATH = Path(__file__).parent.resolve()
runs_base_path = ROOT_PATH / "runs_ppg"
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
    """线性学习率调度器"""
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
        sim_device_id=0,
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
    actor_critic = PPGActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=hidden_dim,
        critic_hidden_dims=hidden_dim,
        activation="mish",
        init_noise_std=init_noise_std,
    ).to(device)

    # 优化器 (分离 Actor 和 Critic)
    actor_params = (
        list(actor_critic.actor_feature_extractor.parameters())
        + list(actor_critic.actor_head.parameters())
        + list(actor_critic.actor_aux_value_head.parameters())
        + [actor_critic.std]
    )
    critic_params = list(actor_critic.critic.parameters())

    actor_optimizer = torch.optim.Adam(actor_params, lr=learning_rate_start)
    critic_optimizer = torch.optim.Adam(critic_params, lr=learning_rate_start)

    # Rollout Buffer
    rollout_buffer = RolloutBuffer(
        num_envs=num_envs,
        n_steps=n_steps,
        obs_dim=obs_dim,
        action_dim=action_dim,
        device=rl_device,
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

    # 统计变量
    rewbuffer = deque(maxlen=100)
    lenbuffer = deque(maxlen=100)
    cur_reward_sum = torch.zeros(num_envs, dtype=torch.float, device=rl_device)
    cur_episode_length = torch.zeros(num_envs, dtype=torch.float, device=rl_device)

    ep_infos = []
    total_steps = 0
    tot_time = 0.0

    print(f"总更新次数: {max_iterations}")

    actor_critic.train()

    for update_idx in track(range(max_iterations), description="训练中"):
        start_time = time.time()

        # 更新学习率 (线性衰减)
        progress = update_idx / max_iterations
        current_lr = linear_schedule(learning_rate_start, learning_rate_end, progress)
        for param_group in actor_optimizer.param_groups:
            param_group["lr"] = current_lr
        for param_group in critic_optimizer.param_groups:
            param_group["lr"] = current_lr

        # ============================================
        # 4.1 收集数据 (Rollout)
        # ============================================
        rollout_buffer.reset()

        with torch.no_grad():
            for step in range(n_steps):
                # 获取动作和 value
                actor_critic.update_distribution(obs)
                actions = actor_critic.distribution.sample()
                log_probs = actor_critic.get_actions_log_prob(actions)
                values = actor_critic.evaluate(obs)

                # 执行动作
                next_obs, privileged_obs, rewards, dones, infos = env.step(actions)

                # 处理超时 (不应视为真正的 done)
                real_dones = dones.clone()
                if "time_outs" in infos:
                    real_dones = dones & ~infos["time_outs"]

                # 存储到 buffer
                rollout_buffer.add(
                    obs, actions, rewards, real_dones.float(), values, log_probs
                )

                # Episode info 收集
                if "episode" in infos:
                    ep_infos.append(infos["episode"])

                # 统计
                cur_reward_sum += rewards
                cur_episode_length += 1
                total_steps += num_envs

                # 处理完成的 episode
                new_ids = (dones > 0).nonzero(as_tuple=False)
                if len(new_ids) > 0:
                    rewbuffer.extend(
                        cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    lenbuffer.extend(
                        cur_episode_length[new_ids][:, 0].cpu().numpy().tolist()
                    )
                    cur_reward_sum[new_ids] = 0
                    cur_episode_length[new_ids] = 0

                obs = next_obs

        collection_time = time.time() - start_time

        # ============================================
        # 4.2 计算 GAE 和 Returns
        # ============================================
        learn_start = time.time()

        with torch.no_grad():
            last_value = actor_critic.evaluate(obs)

        rollout_buffer.compute_gae(last_value, gamma, gae_lambda)

        # 统计 (在更新前)
        adv_mean = rollout_buffer.advantages.mean().item()
        adv_std = rollout_buffer.advantages.std().item()
        adv_min = rollout_buffer.advantages.min().item()
        adv_max = rollout_buffer.advantages.max().item()
        value_mean = rollout_buffer.values.mean().item()
        value_std = rollout_buffer.values.std().item()
        returns_mean = rollout_buffer.returns.mean().item()
        returns_std = rollout_buffer.returns.std().item()

        # Explained Variance
        y_pred = rollout_buffer.values.flatten().cpu().numpy()
        y_true = rollout_buffer.returns.flatten().cpu().numpy()
        var_y = np.var(y_true)
        explained_var = 1 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 0 else 0

        # ============================================
        # 4.3 PPO 更新
        # ============================================
        total_pg_loss = 0
        total_value_loss = 0
        total_entropy_loss = 0
        total_actor_loss = 0
        total_aux_value_loss = 0
        total_aux_kl_loss = 0
        total_loss = 0
        total_kl = 0
        n_updates = 0
        n_aux_updates = 0

        for epoch in range(1, num_learning_epochs + 1):
            for (
                obs_batch,
                actions_batch,
                old_log_probs_batch,
                advantages_batch,
                returns_batch,
            ) in rollout_buffer.get_batches(num_mini_batches):

                # 前向传播
                actor_critic.update_distribution(obs_batch)
                new_log_probs = actor_critic.get_actions_log_prob(actions_batch)
                entropy = actor_critic.entropy.mean()
                values = actor_critic.evaluate(obs_batch).squeeze(-1)

                # 计算 ratio
                ratio = torch.exp(new_log_probs - old_log_probs_batch)

                # Clipped Surrogate Loss
                surr1 = ratio * advantages_batch
                surr2 = (
                    torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param)
                    * advantages_batch
                )
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value Loss
                value_loss = value_loss_coef + nn.functional.mse_loss(
                    values, returns_batch
                )

                # Entropy Loss (鼓励探索)
                entropy_loss = -entropy_coef * entropy

                actor_loss = pg_loss + entropy_loss

                # Total Loss
                loss = pg_loss + value_loss + entropy_loss

                # 反向传播 (分离 Actor 和 Critic 更新)

                critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(critic_params, max_grad_norm)
                critic_optimizer.step()

                actor_optimizer.zero_grad()
                actor_loss.backward()
                nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
                actor_optimizer.step()

                # Aux update
                if epoch % aux_update_freq == 0:
                    for _ in range(n_aux_epochs):
                        new_aux_values = actor_critic.aux_value(obs_batch)

                        aux_value_loss = nn.functional.mse_loss(
                            new_aux_values, returns_batch
                        )
                        loss_kl = nn.functional.kl_div(
                            nn.functional.log_softmax(new_log_probs, dim=-1),
                            nn.functional.log_softmax(old_log_probs_batch, dim=-1),
                            reduction="batchmean",
                        )

                        aux_total_loss = aux_value_loss + beta_clone * loss_kl

                        actor_optimizer.zero_grad()
                        aux_total_loss.backward()
                        nn.utils.clip_grad_norm_(actor_params, max_grad_norm)
                        actor_optimizer.step()

                        total_aux_value_loss += aux_value_loss.item()
                        total_aux_kl_loss += loss_kl.item()
                        n_aux_updates += 1

                # 计算 KL 散度 (近似 KL)
                with torch.no_grad():
                    approx_kl = (old_log_probs_batch - new_log_probs).mean().item()

                total_pg_loss += pg_loss.item()
                total_value_loss += value_loss.item()
                total_entropy_loss += entropy_loss.item()
                total_actor_loss += actor_loss.item()
                total_loss += loss.item()
                total_kl += approx_kl
                n_updates += 1

        # 平均 loss
        mean_pg_loss = total_pg_loss / n_updates
        mean_value_loss = total_value_loss / n_updates
        mean_entropy_loss = total_entropy_loss / n_updates
        mean_actor_loss = total_actor_loss / n_updates
        mean_total_loss = total_loss / n_updates
        mean_kl = total_kl / n_updates

        # PPG aux loss
        mean_aux_value_loss = (
            total_aux_value_loss / n_aux_updates if n_aux_updates > 0 else 0
        )
        mean_aux_kl_loss = total_aux_kl_loss / n_aux_updates if n_aux_updates > 0 else 0

        learn_time = time.time() - learn_start
        tot_time += collection_time + learn_time

        # ============================================
        # 4.4 计算额外指标
        # ============================================
        mean_std = actor_critic.std.mean().item()

        # ============================================
        # 4.5 记录到 TensorBoard
        # ============================================
        it = update_idx

        # Loss
        writer.add_scalar("Loss/policy", mean_pg_loss, it)
        writer.add_scalar("Loss/value", mean_value_loss, it)
        writer.add_scalar("Loss/entropy", mean_entropy_loss, it)
        writer.add_scalar("Loss/actor", mean_actor_loss, it)
        writer.add_scalar("Loss/total", mean_total_loss, it)

        # PPG Aux Loss
        writer.add_scalar("Loss/aux_value", mean_aux_value_loss, it)
        writer.add_scalar("Loss/aux_kl", mean_aux_kl_loss, it)

        # Learning Rate
        writer.add_scalar("Loss/learning_rate", current_lr, it)

        # Policy / Exploration
        writer.add_scalar("Policy/mean_noise_std", mean_std, it)
        writer.add_scalar("Policy/kl_divergence", mean_kl, it)
        writer.add_scalar("Policy/entropy", -mean_entropy_loss, it)

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

        # Episode Info
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
                writer.add_scalar("Episode/" + key, value, it)

        # Train metrics
        if len(rewbuffer) > 0:
            mean_reward = statistics.mean(rewbuffer)
            mean_ep_len = statistics.mean(lenbuffer)
            std_reward = statistics.stdev(rewbuffer) if len(rewbuffer) > 1 else 0
            std_ep_len = statistics.stdev(lenbuffer) if len(lenbuffer) > 1 else 0

            writer.add_scalar("Train/mean_reward", mean_reward, it)
            writer.add_scalar("Train/std_reward", std_reward, it)
            writer.add_scalar("Train/mean_episode_length", mean_ep_len, it)
            writer.add_scalar("Train/std_episode_length", std_ep_len, it)
            writer.add_scalar("Train/mean_reward/time", mean_reward, tot_time)
            writer.add_scalar("Train/mean_episode_length/time", mean_ep_len, tot_time)

        # ============================================
        # 4.6 定期打印统计信息
        # ============================================
        if update_idx % log_freq == 0:
            width = 80
            pad = 35

            log_string = f"\n{'#' * width}\n"
            log_string += (
                f" Learning iteration {update_idx}/{max_iterations} ".center(width, " ")
                + "\n\n"
            )
            log_string += f"{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {collection_time:.3f}s, learning: {learn_time:.3f}s)\n"
            log_string += f"{'Policy loss:':>{pad}} {mean_pg_loss:.4f}\n"
            log_string += f"{'Value loss:':>{pad}} {mean_value_loss:.4f}\n"
            log_string += f"{'Actor loss:':>{pad}} {mean_actor_loss:.4f}\n"
            log_string += f"{'Entropy:':>{pad}} {-mean_entropy_loss:.4f}\n"
            log_string += f"{'Aux value loss:':>{pad}} {mean_aux_value_loss:.4f}\n"
            log_string += f"{'Aux KL loss:':>{pad}} {mean_aux_kl_loss:.4f}\n"
            log_string += f"{'Mean action noise std:':>{pad}} {mean_std:.4f}\n"
            log_string += f"{'KL divergence:':>{pad}} {mean_kl:.6f}\n"
            log_string += f"{'Learning rate:':>{pad}} {current_lr:.2e}\n"
            log_string += f"{'Explained Variance:':>{pad}} {explained_var:.4f}\n"
            log_string += (
                f"{'Value (mean/std):':>{pad}} {value_mean:.2f} / {value_std:.2f}\n"
            )
            log_string += (
                f"{'Advantage (mean/std):':>{pad}} {adv_mean:.4f} / {adv_std:.4f}\n"
            )
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
            eta = (
                tot_time / (update_idx + 1) * (max_iterations - update_idx)
                if update_idx > 0
                else 0
            )
            log_string += f"{'ETA:':>{pad}} {eta:.1f}s\n"

            print(log_string)

        # 清空 episode info buffer
        ep_infos.clear()

        # ============================================
        # 4.7 保存模型
        # ============================================
        if update_idx > 0 and update_idx % save_freq == 0:
            checkpoint_path = model_save_path / "checkpoints"
            os.makedirs(str(checkpoint_path), exist_ok=True)
            torch.save(
                {
                    "update_idx": update_idx,
                    "total_steps": total_steps,
                    "actor_critic": actor_critic.state_dict(),
                    "actor_optimizer": actor_optimizer.state_dict(),
                    "critic_optimizer": critic_optimizer.state_dict(),
                },
                checkpoint_path / f"checkpoint_{update_idx}.pt",
            )
            print(f"模型已保存: {checkpoint_path}/checkpoint_{update_idx}.pt")

    # 保存最终模型
    torch.save(
        {
            "actor_critic": actor_critic.state_dict(),
            "actor_optimizer": actor_optimizer.state_dict(),
            "critic_optimizer": critic_optimizer.state_dict(),
        },
        model_save_path / "final_model.pt",
    )

    writer.close()
    print(f"\n训练完成! 模型保存在: {model_save_path}")


if __name__ == "__main__":
    main()
