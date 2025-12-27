"""
通用评估脚本 - 支持 PPO 和 PPG checkpoints

Usage examples:
  python mytrain/eval_ppo.py --checkpoint runs_ppo/run1/final_model.pt --num_envs 16
  python mytrain/eval_ppo.py -c runs_ppo/run1/checkpoints/checkpoint_500.pt --render
  python mytrain/eval_ppo.py -c runs_ppg/run1/final_model.pt --model_type ppg

Options:
  --checkpoint   Path to checkpoint (contains 'actor_critic' state_dict).
  --task         Environment task name (default: go2).
  --num_envs     Number of parallel envs to create for evaluation (default: 16).
  --hidden_dims  Comma-separated hidden dims used by actor/critic (default: 512,512,512).
  --sim_device   Simulation device (cpu or cuda:0). Default uses cuda:0.
  --steps        Number of steps to run (default: env.max_episode_length * 5).
  --render       Enable rendering (opens a viewer window).
  --deterministic Use deterministic actions (no noise).
  --model_type   Model type: 'ppo' or 'ppg' (default: auto-detect from checkpoint).

The script will create the env, load the actor_critic weights and run several episodes printing per-episode rewards.
"""

import os
import sys
from pathlib import Path
import argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import task_registry
import torch

from models import PPOActorCritic, PPGActorCritic  # 导入两种 ActorCritic


def parse_hidden(hidden_str):
    """Parse comma-separated hidden dims string to list."""
    parts = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]
    return parts


def make_args(task, num_envs, sim_device, rl_device, headless=True):
    """Build a Namespace similar to ppo_legged's args used by task_registry.make_env."""
    return SimpleNamespace(
        task=task,
        num_envs=num_envs,
        headless=headless,
        sim_device=sim_device,
        rl_device=rl_device,
        physics_engine=gymapi.SIM_PHYSX,
        use_gpu=(sim_device != "cpu"),
        use_gpu_pipeline=(sim_device != "cpu"),
        subscenes=4,
        num_threads=0,
        device=rl_device,
        compute_device_id=0,
        sim_device_type="cuda" if sim_device != "cpu" else "cpu",
        sim_device_id=0,
    )


def detect_model_type(state_dict):
    """
    自动检测模型类型 (PPO 或 PPG)

    PPO 模型包含 'actor.' 前缀的参数
    PPG 模型包含 'actor_feature_extractor.', 'actor_head.', 'actor_aux_value_head.' 前缀的参数
    """
    keys = list(state_dict.keys())

    # 检查是否有 PPG 特有的参数
    has_ppg_keys = any(
        "actor_feature_extractor" in k or "actor_aux_value_head" in k for k in keys
    )
    has_ppo_keys = any(k.startswith("actor.") for k in keys)

    if has_ppg_keys:
        return "ppg"
    elif has_ppo_keys:
        return "ppo"
    else:
        # 默认使用 PPO
        print("警告: 无法自动检测模型类型，默认使用 PPO")
        return "ppo"


def create_actor_critic(
    model_type, obs_dim, action_dim, hidden_dims, init_noise_std, rl_device
):
    """根据模型类型创建对应的 ActorCritic 网络"""
    if model_type == "ppg":
        print(f"使用 PPG 模型 (PPGActorCritic)")
        actor_critic = PPGActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
            activation="mish",
            init_noise_std=init_noise_std,
        ).to(rl_device)
    else:
        print(f"使用 PPO 模型 (PPOActorCritic)")
        actor_critic = PPOActorCritic(
            num_actor_obs=obs_dim,
            num_critic_obs=obs_dim,
            num_actions=action_dim,
            actor_hidden_dims=hidden_dims,
            critic_hidden_dims=hidden_dims,
            activation="mish",
            init_noise_std=init_noise_std,
        ).to(rl_device)

    return actor_critic


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO/PPG checkpoints")
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to checkpoint (.pt)"
    )
    parser.add_argument("--task", type=str, default="go2", help="Environment task name")
    parser.add_argument(
        "--num_envs",
        type=int,
        default=16,
        help="Number of parallel envs for evaluation",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="512,512,512",
        help="Comma-separated hidden dims for actor/critic networks",
    )
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cuda:0",
        help="Simulation device, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--rl_device", type=str, default="cuda:0", help="RL device for tensors"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Number of environment steps to run (0 -> uses 5 episodes length)",
    )
    parser.add_argument(
        "--render", action="store_true", help="Enable rendering (opens a viewer window)"
    )
    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Use deterministic actions (no noise)",
    )
    parser.add_argument(
        "--init_noise_std",
        type=float,
        default=1.0,
        help="Initial noise std (should match training config)",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "ppo", "ppg"],
        help="Model type: ppo, ppg, or auto (auto-detect from checkpoint)",
    )
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    hidden_dims = parse_hidden(args.hidden_dims)

    # Load checkpoint first to detect model type
    print(f"\n加载 checkpoint: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")

    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        state = ckpt["actor_critic"]
    else:
        # Maybe it's the state_dict directly
        state = ckpt

    # Detect or use specified model type
    if args.model_type == "auto":
        model_type = detect_model_type(state)
        print(f"自动检测模型类型: {model_type.upper()}")
    else:
        model_type = args.model_type
        print(f"使用指定模型类型: {model_type.upper()}")

    # Create env
    print(f"\n创建环境: task={args.task}, num_envs={args.num_envs}")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs

    headless = not args.render
    makeenv_args = make_args(
        args.task, args.num_envs, args.sim_device, args.rl_device, headless=headless
    )
    env, _ = task_registry.make_env(name=args.task, args=makeenv_args, env_cfg=env_cfg)

    obs = env.get_observations()
    obs_dim = env.num_obs
    action_dim = env.num_actions
    rl_device = args.rl_device

    print(f"环境创建成功: obs_dim={obs_dim}, action_dim={action_dim}")

    # Instantiate ActorCritic based on model type
    actor_critic = create_actor_critic(
        model_type=model_type,
        obs_dim=obs_dim,
        action_dim=action_dim,
        hidden_dims=hidden_dims,
        init_noise_std=args.init_noise_std,
        rl_device=rl_device,
    )

    # Load checkpoint weights
    try:
        actor_critic.load_state_dict(state)
        print("模型加载成功!")
    except Exception as e:
        print("Failed to load actor_critic state_dict.")
        print(
            "This usually means the network architecture (hidden dims) does not match the checkpoint."
        )
        print("Error:", e)
        print()
        print("提示:")
        print("  1. 检查 --hidden_dims 是否与训练时一致 (例如 512,512,512)")
        print("  2. 检查 --model_type 是否正确 (ppo 或 ppg)")
        print("  3. 如果自动检测失败，请手动指定 --model_type")
        return

    actor_critic.eval()

    # Run episodes
    max_episode_length = env.max_episode_length
    total_steps = args.steps if args.steps > 0 else int(max_episode_length * 5)

    episode_rewards = torch.zeros(args.num_envs, device=rl_device)
    episode_lengths = torch.zeros(args.num_envs, device=rl_device)
    completed_episodes = 0
    all_rewards = []

    obs = env.get_observations()

    # Resample commands
    env._resample_commands(torch.arange(args.num_envs, device=rl_device))
    obs = env.get_observations()

    print(f"\n开始评估, 总步数: {total_steps}")
    print(f"模型类型: {model_type.upper()}")
    print(f"{'Deterministic' if args.deterministic else 'Stochastic'} actions")
    print("-" * 60)

    for step in range(total_steps):
        obs_tensor = obs.to(rl_device)

        with torch.no_grad():
            if args.deterministic:
                # Deterministic: use mean action only
                actions = actor_critic.act_inference(obs_tensor)
            else:
                # Stochastic: sample from distribution
                actions = actor_critic.act(obs_tensor)

        next_obs, privileged_obs, rewards, dones, infos = env.step(actions)

        episode_rewards += rewards.to(rl_device)
        episode_lengths += 1

        # Handle completed episodes
        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            for did in done_ids.tolist():
                ep_reward = float(episode_rewards[did].item())
                ep_length = int(episode_lengths[did].item())
                all_rewards.append(ep_reward)
                print(
                    f"Episode完成 (env={did:2d}) | 长度: {ep_length:4d} | 奖励: {ep_reward:8.2f}"
                )

            completed_episodes += len(done_ids)
            episode_rewards[done_ids] = 0
            episode_lengths[done_ids] = 0

        obs = next_obs

    # Summary statistics
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
    print(f"模型类型: {model_type.upper()}")
    print(f"完成的Episode数: {completed_episodes}")

    if all_rewards:
        import statistics

        mean_reward = statistics.mean(all_rewards)
        std_reward = statistics.stdev(all_rewards) if len(all_rewards) > 1 else 0
        min_reward = min(all_rewards)
        max_reward = max(all_rewards)

        print(f"平均奖励: {mean_reward:.2f} ± {std_reward:.2f}")
        print(f"奖励范围: [{min_reward:.2f}, {max_reward:.2f}]")
    else:
        print("没有完成任何完整的episode")

    print("=" * 60)


if __name__ == "__main__":
    main()
