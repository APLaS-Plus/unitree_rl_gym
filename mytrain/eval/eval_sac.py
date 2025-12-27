"""
Simple evaluation script for SAC checkpoints saved by `mytrain/sac_legged.py`.

Usage examples:
  python mytrain/eval_sac.py --task go2 --checkpoint runs_sac/run1/final_model.pt --num_envs 16

Options:
  --checkpoint   Path to checkpoint (contains 'actor' state_dict).
  --task         Environment task name (default: go2).
  --num_envs     Number of parallel envs to create for evaluation (default: 16).
  --hidden_dims  Comma-separated hidden dims used by the actor (default: 512,512,256).
  --sim_device   Simulation device (cpu or cuda:0). Default uses cpu for safety.
  --steps        Number of steps to run (default: env.max_episode_length * 5).

The script will create the env, load the actor weights and run several episodes printing per-episode rewards.
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
from models import StochasticActor, device as model_device


def parse_hidden(hidden_str):
    parts = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]
    return tuple(parts)


def make_args(task, num_envs, sim_device, rl_device, headless=True):
    # Build a Namespace similar to sac_legged's args used by task_registry.make_env
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to checkpoint (.pt)"
    )
    parser.add_argument("--task", type=str, default="go2")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="512,512,512",
        help="Comma-separated hidden dims",
    )
    parser.add_argument(
        "--sim_device",
        type=str,
        default="cuda:0",
        help="Simulation device, e.g. cpu or cuda:0",
    )
    parser.add_argument(
        "--rl_device",
        type=str,
        default=model_device,
        help="RL device for tensors (default from models.py)",
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
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    hidden_dims = parse_hidden(args.hidden_dims)

    # Create env
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

    # Determine rl_device early so we can place tensors correctly
    rl_device = torch.device(
        args.rl_device
        if args.rl_device.startswith("cpu") or args.rl_device.startswith("cuda")
        else args.rl_device
    )
    action_high = torch.ones(action_dim, device=rl_device)

    print(
        f"Env created: task={args.task}, num_envs={args.num_envs}, obs_dim={obs_dim}, action_dim={action_dim}"
    )

    # Instantiate actor with the same architecture used in training
    actor = StochasticActor(obs_dim, action_dim, action_high, hidden_dim=hidden_dims)

    # Load checkpoint
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "actor" in ckpt:
        state = ckpt["actor"]
    else:
        state = ckpt

    try:
        actor.load_state_dict(state)
    except Exception as e:
        print(
            "Failed to load actor state_dict. This usually means the network architecture (hidden dims) does not match the checkpoint."
        )
        print("Error:", e)
        print(
            "Try specifying --hidden_dims to match the training hidden sizes (e.g. 512,512,256)."
        )
        return

    # Move actor to device
    actor.to(rl_device)
    actor.eval()

    # Run episodes
    max_episode_length = env.max_episode_length
    total_steps = args.steps if args.steps > 0 else int(max_episode_length * 5)

    episode_rewards = torch.zeros(args.num_envs, device=rl_device)
    episode_lengths = torch.zeros(args.num_envs, device=rl_device)
    completed = 0

    obs = env.get_observations()

    for step in range(total_steps):
        # Ensure obs on correct device and dtype
        obs_tensor = obs.to(rl_device)
        with torch.no_grad():
            actions, _ = actor.sample(obs_tensor)

        next_obs, privileged_obs, rewards, dones, infos = env.step(actions)

        episode_rewards += rewards.to(rl_device)
        episode_lengths += 1

        done_ids = dones.nonzero(as_tuple=False).flatten()
        if len(done_ids) > 0:
            for did in done_ids.tolist():
                print(
                    f"Episode finished (env={did}) length={int(episode_lengths[did].item())} reward={float(episode_rewards[did].item()):.3f}"
                )
            completed += len(done_ids)
            episode_rewards[done_ids] = 0
            episode_lengths[done_ids] = 0

        obs = next_obs

    print(f"Finished evaluation run. Completed episodes: {completed}")


if __name__ == "__main__":
    main()
