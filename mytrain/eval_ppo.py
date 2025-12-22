"""
Simple evaluation script for PPO checkpoints saved by `mytrain/ppo_legged.py`.

Usage examples:
  python mytrain/eval_ppo.py --checkpoint runs_ppo/run1/final_model.pt --num_envs 16
  python mytrain/eval_ppo.py -c runs_ppo/run1/checkpoints/checkpoint_500.pt --render

Options:
  --checkpoint   Path to checkpoint (contains 'actor_critic' state_dict).
  --task         Environment task name (default: go2).
  --num_envs     Number of parallel envs to create for evaluation (default: 16).
  --hidden_dims  Comma-separated hidden dims used by actor/critic (default: 512,512,256).
  --sim_device   Simulation device (cpu or cuda:0). Default uses cuda:0.
  --steps        Number of steps to run (default: env.max_episode_length * 5).
  --render       Enable rendering (opens a viewer window).
  --deterministic Use deterministic actions (no noise).

The script will create the env, load the actor_critic weights and run several episodes printing per-episode rewards.
"""

import os
import sys
from pathlib import Path
import argparse
from types import SimpleNamespace

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import task_registry
import torch

from models import PPOActorCritic  # 使用自定义 ActorCritic


def parse_hidden(hidden_str):
    """Parse comma-separated hidden dims string to list."""
    parts = [int(x.strip()) for x in hidden_str.split(',') if x.strip()]
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
        use_gpu=(sim_device != 'cpu'),
        use_gpu_pipeline=(sim_device != 'cpu'),
        subscenes=4,
        num_threads=0,
        device=rl_device,
        compute_device_id=0,
        sim_device_type='cuda' if sim_device != 'cpu' else 'cpu',
        sim_device_id=0,
    )


def main():
    parser = argparse.ArgumentParser(description="Evaluate PPO checkpoints")
    parser.add_argument('--checkpoint', '-c', type=str, required=True, 
                        help='Path to checkpoint (.pt)')
    parser.add_argument('--task', type=str, default='go2',
                        help='Environment task name')
    parser.add_argument('--num_envs', type=int, default=16,
                        help='Number of parallel envs for evaluation')
    parser.add_argument('--hidden_dims', type=str, default='512,512,256', 
                        help='Comma-separated hidden dims for actor/critic networks')
    parser.add_argument('--sim_device', type=str, default='cuda:0', 
                        help='Simulation device, e.g. cpu or cuda:0')
    parser.add_argument('--rl_device', type=str, default='cuda:0', 
                        help='RL device for tensors')
    parser.add_argument('--steps', type=int, default=0, 
                        help='Number of environment steps to run (0 -> uses 5 episodes length)')
    parser.add_argument('--render', action='store_true', 
                        help='Enable rendering (opens a viewer window)')
    parser.add_argument('--deterministic', action='store_true',
                        help='Use deterministic actions (no noise)')
    parser.add_argument('--init_noise_std', type=float, default=1.0,
                        help='Initial noise std (should match training config)')
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    hidden_dims = parse_hidden(args.hidden_dims)

    # Create env
    print(f"\n创建环境: task={args.task}, num_envs={args.num_envs}")
    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)
    env_cfg.env.num_envs = args.num_envs

    headless = not args.render
    makeenv_args = make_args(args.task, args.num_envs, args.sim_device, args.rl_device, headless=headless)
    env, _ = task_registry.make_env(name=args.task, args=makeenv_args, env_cfg=env_cfg)

    obs = env.get_observations()
    obs_dim = env.num_obs
    action_dim = env.num_actions
    rl_device = args.rl_device

    print(f"环境创建成功: obs_dim={obs_dim}, action_dim={action_dim}")

    # Instantiate PPOActorCritic with the same architecture used in training
    actor_critic = PPOActorCritic(
        num_actor_obs=obs_dim,
        num_critic_obs=obs_dim,
        num_actions=action_dim,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation='elu',
        init_noise_std=args.init_noise_std,
    ).to(rl_device)

    # Load checkpoint
    print(f"加载模型: {checkpoint_path}")
    ckpt = torch.load(str(checkpoint_path), map_location='cpu')
    
    if isinstance(ckpt, dict) and 'actor_critic' in ckpt:
        state = ckpt['actor_critic']
    else:
        # Maybe it's the state_dict directly
        state = ckpt

    try:
        actor_critic.load_state_dict(state)
        print("模型加载成功!")
    except Exception as e:
        print('Failed to load actor_critic state_dict.')
        print('This usually means the network architecture (hidden dims) does not match the checkpoint.')
        print('Error:', e)
        print('Try specifying --hidden_dims to match the training hidden sizes (e.g. 512,512,256).')
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
                print(f"Episode完成 (env={did:2d}) | 长度: {ep_length:4d} | 奖励: {ep_reward:8.2f}")
            
            completed_episodes += len(done_ids)
            episode_rewards[done_ids] = 0
            episode_lengths[done_ids] = 0

        obs = next_obs

    # Summary statistics
    print("\n" + "=" * 60)
    print("评估结果汇总")
    print("=" * 60)
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


if __name__ == '__main__':
    main()
