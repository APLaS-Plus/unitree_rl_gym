"""
Export trained PPO/PPG models to TorchScript format for MuJoCo deployment.

Usage:
    python mytrain/eval/export_jit.py -c mytrain/runs_ppo/run2/final_model.pt -o policy.pt
    python mytrain/eval/export_jit.py -c mytrain/runs_ppg/run4/final_model.pt -o policy.pt --model_type ppg

Options:
    -c, --checkpoint  Path to training checkpoint (.pt)
    -o, --output      Output path for TorchScript model (default: policy_jit.pt)
    --model_type      Model type: 'auto', 'ppo', or 'ppg' (default: auto)
    --hidden_dims     Hidden dimensions (default: 512,512,512)
    --num_obs         Number of observations (default: 48 for Go2)
    --num_actions     Number of actions (default: 12 for Go2)
"""

import os
import sys
from pathlib import Path
import argparse

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn as nn

from models import PPOActorCritic, PPGActorCritic


def parse_hidden(hidden_str):
    """Parse comma-separated hidden dims string to list."""
    parts = [int(x.strip()) for x in hidden_str.split(",") if x.strip()]
    return parts


def detect_model_type(state_dict):
    """
    Auto-detect model type (PPO or PPG) from state dict keys.
    """
    keys = list(state_dict.keys())
    has_ppg_keys = any(
        "actor_feature_extractor" in k or "actor_aux_value_head" in k for k in keys
    )
    has_ppo_keys = any(k.startswith("actor.") for k in keys)

    if has_ppg_keys:
        return "ppg"
    elif has_ppo_keys:
        return "ppo"
    else:
        print("Warning: Cannot auto-detect model type, defaulting to PPO")
        return "ppo"


class PPOActorJIT(nn.Module):
    """
    JIT-compatible actor module for PPO.
    Only contains the actor network for inference (no critic, no std).
    """

    def __init__(self, actor: nn.Sequential):
        super().__init__()
        self.actor = actor

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.actor(obs)


class PPGActorJIT(nn.Module):
    """
    JIT-compatible actor module for PPG.
    Only contains the actor feature extractor and head for inference.
    """

    def __init__(self, feature_extractor: nn.Sequential, head: nn.Linear):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.head = head

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        features = self.feature_extractor(obs)
        return self.head(features)


def export_ppo_actor(
    checkpoint_path: Path,
    output_path: Path,
    hidden_dims: list,
    num_obs: int,
    num_actions: int,
):
    """Export PPO actor to TorchScript."""
    print(f"Loading PPO checkpoint: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        state = ckpt["actor_critic"]
    else:
        state = ckpt

    # Create actor-critic model
    actor_critic = PPOActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="mish",
        init_noise_std=1.0,
    )

    # Load weights
    actor_critic.load_state_dict(state)
    actor_critic.eval()
    print("Model loaded successfully!")

    # Create JIT-compatible actor
    actor_jit = PPOActorJIT(actor_critic.actor)
    actor_jit.eval()

    # Trace the model
    dummy_input = torch.randn(1, num_obs)
    traced_model = torch.jit.trace(actor_jit, dummy_input)

    # Verify output
    with torch.no_grad():
        original_output = actor_critic.act_inference(dummy_input)
        traced_output = traced_model(dummy_input)

    diff = (original_output - traced_output).abs().max().item()
    print(f"Max output difference: {diff:.6e}")

    if diff > 1e-5:
        print("Warning: Output difference is larger than expected!")

    # Save
    traced_model.save(str(output_path))
    print(f"\nTorchScript model saved to: {output_path}")

    return traced_model


def export_ppg_actor(
    checkpoint_path: Path,
    output_path: Path,
    hidden_dims: list,
    num_obs: int,
    num_actions: int,
):
    """Export PPG actor to TorchScript."""
    print(f"Loading PPG checkpoint: {checkpoint_path}")

    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        state = ckpt["actor_critic"]
    else:
        state = ckpt

    # Create actor-critic model
    actor_critic = PPGActorCritic(
        num_actor_obs=num_obs,
        num_critic_obs=num_obs,
        num_actions=num_actions,
        actor_hidden_dims=hidden_dims,
        critic_hidden_dims=hidden_dims,
        activation="mish",
        init_noise_std=1.0,
    )

    # Load weights
    actor_critic.load_state_dict(state)
    actor_critic.eval()
    print("Model loaded successfully!")

    # Create JIT-compatible actor
    actor_jit = PPGActorJIT(
        actor_critic.actor_feature_extractor, actor_critic.actor_head
    )
    actor_jit.eval()

    # Trace the model
    dummy_input = torch.randn(1, num_obs)
    traced_model = torch.jit.trace(actor_jit, dummy_input)

    # Verify output
    with torch.no_grad():
        original_output = actor_critic.act_inference(dummy_input)
        traced_output = traced_model(dummy_input)

    diff = (original_output - traced_output).abs().max().item()
    print(f"Max output difference: {diff:.6e}")

    if diff > 1e-5:
        print("Warning: Output difference is larger than expected!")

    # Save
    traced_model.save(str(output_path))
    print(f"\nTorchScript model saved to: {output_path}")

    return traced_model


def main():
    parser = argparse.ArgumentParser(description="Export PPO/PPG model to TorchScript")
    parser.add_argument(
        "-c",
        "--checkpoint",
        type=str,
        required=True,
        help="Path to training checkpoint (.pt)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="policy_jit.pt",
        help="Output path for TorchScript model",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="auto",
        choices=["auto", "ppo", "ppg"],
        help="Model type: auto, ppo, or ppg",
    )
    parser.add_argument(
        "--hidden_dims",
        type=str,
        default="512,512,512",
        help="Comma-separated hidden dimensions",
    )
    parser.add_argument(
        "--num_obs", type=int, default=48, help="Number of observations (48 for Go2)"
    )
    parser.add_argument(
        "--num_actions", type=int, default=12, help="Number of actions (12 for Go2)"
    )

    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    assert checkpoint_path.exists(), f"Checkpoint not found: {checkpoint_path}"

    # Determine output path
    if args.output == "policy_jit.pt":
        output_path = checkpoint_path.parent / "policy_jit.pt"
    else:
        output_path = Path(args.output)

    hidden_dims = parse_hidden(args.hidden_dims)

    print("=" * 60)
    print("Export PPO/PPG Model to TorchScript")
    print("=" * 60)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Output: {output_path}")
    print(f"Hidden dims: {hidden_dims}")
    print(f"Observations: {args.num_obs}, Actions: {args.num_actions}")
    print()

    # Detect or use specified model type
    ckpt = torch.load(str(checkpoint_path), map_location="cpu")
    if isinstance(ckpt, dict) and "actor_critic" in ckpt:
        state = ckpt["actor_critic"]
    else:
        state = ckpt

    if args.model_type == "auto":
        model_type = detect_model_type(state)
        print(f"Auto-detected model type: {model_type.upper()}")
    else:
        model_type = args.model_type
        print(f"Using specified model type: {model_type.upper()}")
    print()

    # Export
    if model_type == "ppg":
        export_ppg_actor(
            checkpoint_path, output_path, hidden_dims, args.num_obs, args.num_actions
        )
    else:
        export_ppo_actor(
            checkpoint_path, output_path, hidden_dims, args.num_obs, args.num_actions
        )

    print("\nExport completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
