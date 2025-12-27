"""
Training Metrics Visualization Script

Plots policy entropy and episode length during training.
Data is loaded from TensorBoard event files.

Usage:
    python mytrain/draw/plot_training_metrics.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorboard.backend.event_processing import event_accumulator

# ============================================
# Configuration
# ============================================

ROOT_PATH = Path(__file__).parent.parent.resolve()  # mytrain/

RUNS_CONFIG = {
    "PPO": ROOT_PATH / "runs_ppo" / "tensorboard" / "run2",
    "PPG": ROOT_PATH / "runs_ppg" / "tensorboard" / "run4",
    "SAC": ROOT_PATH / "runs_sac" / "tensorboard" / "run9",
}

OUTPUT_DIR = Path(__file__).parent.resolve()  # draw/

FIGURE_SIZE = (14, 6)
DPI = 150
SMOOTHING_WINDOW = 50

# Colors
COLORS = {
    "PPO": "#2196F3",  # Blue
    "PPG": "#4CAF50",  # Green
    "SAC": "#FF5722",  # Orange
}


def load_tensorboard_data(log_dir: Path, tag: str) -> pd.DataFrame:
    """Load scalar data from TensorBoard event files."""
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={event_accumulator.SCALARS: 0},
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        return pd.DataFrame()

    events = ea.Scalars(tag)
    data = {
        "step": [e.step for e in events],
        "value": [e.value for e in events],
        "wall_time": [e.wall_time for e in events],
    }

    df = pd.DataFrame(data)
    if len(df) > 0:
        df["time"] = df["wall_time"] - df["wall_time"].iloc[0]
        df["time_minutes"] = df["time"] / 60

    return df


def smooth_curve(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply rolling mean smoothing."""
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def plot_entropy_and_episode_length():
    """Plot entropy and episode length vs training time."""
    print("Loading TensorBoard data...")

    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 11

    # Entropy tags differ by algorithm
    entropy_tags = {
        "PPO": "Policy/entropy",
        "PPG": "Policy/entropy",
        "SAC": "Entropy/policy_entropy",
    }

    # Episode length tags
    episode_length_tags = {
        "PPO": "Train/mean_episode_length",
        "PPG": "Train/mean_episode_length",
        "SAC": "Train/completed_episodes",  # SAC logs completed episodes per step
    }

    fig, axes = plt.subplots(1, 2, figsize=FIGURE_SIZE)

    # ==================== Plot 1: Entropy ====================
    ax1 = axes[0]
    entropy_data = {}

    for algo_name, log_dir in RUNS_CONFIG.items():
        if not log_dir.exists():
            continue

        tag = entropy_tags[algo_name]
        df = load_tensorboard_data(log_dir, tag)

        if df.empty:
            print(f"Warning: No entropy data for {algo_name}")
            continue

        entropy_data[algo_name] = df
        smoothed = smooth_curve(df["value"].values, SMOOTHING_WINDOW)

        ax1.plot(
            df["time_minutes"],
            smoothed,
            label=algo_name,
            color=COLORS[algo_name],
            linewidth=2,
            alpha=0.9,
        )

        ax1.fill_between(
            df["time_minutes"],
            smooth_curve(df["value"].values, SMOOTHING_WINDOW // 2)
            - df["value"].std() * 0.2,
            smooth_curve(df["value"].values, SMOOTHING_WINDOW // 2)
            + df["value"].std() * 0.2,
            alpha=0.15,
            color=COLORS[algo_name],
        )

        print(
            f"Loaded {algo_name} entropy: {len(df)} points, "
            f"range: {df['value'].min():.3f} - {df['value'].max():.3f}"
        )

    ax1.set_xlabel("Training Time (minutes)", fontweight="bold")
    ax1.set_ylabel("Policy Entropy", fontweight="bold")
    ax1.set_title("Policy Entropy During Training", fontweight="bold")
    ax1.legend(loc="best", frameon=True)
    ax1.grid(True, alpha=0.3)

    # ==================== Plot 2: Episode Length ====================
    ax2 = axes[1]
    ep_len_data = {}

    for algo_name, log_dir in RUNS_CONFIG.items():
        if not log_dir.exists():
            continue

        tag = episode_length_tags[algo_name]
        if tag is None:
            continue

        df = load_tensorboard_data(log_dir, tag)

        if df.empty:
            print(f"Warning: No episode length data for {algo_name}")
            continue

        ep_len_data[algo_name] = df
        smoothed = smooth_curve(df["value"].values, SMOOTHING_WINDOW)

        ax2.plot(
            df["time_minutes"],
            smoothed,
            label=algo_name,
            color=COLORS[algo_name],
            linewidth=2,
            alpha=0.9,
        )

        ax2.fill_between(
            df["time_minutes"],
            smooth_curve(df["value"].values, SMOOTHING_WINDOW // 2)
            - df["value"].std() * 0.2,
            smooth_curve(df["value"].values, SMOOTHING_WINDOW // 2)
            + df["value"].std() * 0.2,
            alpha=0.15,
            color=COLORS[algo_name],
        )

        print(
            f"Loaded {algo_name} episode length: {len(df)} points, "
            f"range: {df['value'].min():.1f} - {df['value'].max():.1f}"
        )

    ax2.set_xlabel("Training Time (minutes)", fontweight="bold")
    ax2.set_ylabel("Mean Episode Length (steps)", fontweight="bold")
    ax2.set_title("Episode Length During Training", fontweight="bold")
    ax2.legend(loc="best", frameon=True)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save
    output_path = OUTPUT_DIR / "training_entropy_and_episode_length.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to: {output_path}")

    pdf_path = OUTPUT_DIR / "training_entropy_and_episode_length.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"PDF saved to: {pdf_path}")

    plt.show()

    # Print summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    for algo_name in RUNS_CONFIG.keys():
        print(f"\n{algo_name}:")
        if algo_name in entropy_data:
            df = entropy_data[algo_name]
            print(
                f"  Entropy: initial={df['value'].iloc[0]:.3f}, "
                f"final={df['value'].iloc[-100:].mean():.3f}"
            )
        if algo_name in ep_len_data:
            df = ep_len_data[algo_name]
            print(
                f"  Episode Length: max={df['value'].max():.1f}, "
                f"final={df['value'].iloc[-100:].mean():.1f}"
            )


def main():
    print("=" * 60)
    print("Training Metrics Visualization")
    print("=" * 60)
    print("\nConfiguration:")
    for algo, path in RUNS_CONFIG.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {algo}: {path} [{exists}]")
    print()

    plot_entropy_and_episode_length()


if __name__ == "__main__":
    main()
