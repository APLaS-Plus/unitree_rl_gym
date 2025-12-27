"""
Training Results Visualization Script

Plots training time vs mean reward for PPO, PPG, and SAC algorithms.
Data is loaded from TensorBoard event files.

Usage:
    python mytrain/draw/plot_training_curves.py
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

# Base path
ROOT_PATH = Path(__file__).parent.parent.resolve()  # mytrain/

# TensorBoard log directories
RUNS_CONFIG = {
    "PPO": ROOT_PATH / "runs_ppo" / "tensorboard" / "run2",
    "PPG": ROOT_PATH / "runs_ppg" / "tensorboard" / "run4",
    "SAC": ROOT_PATH / "runs_sac" / "tensorboard" / "run9",
}

# Output directory
OUTPUT_DIR = Path(__file__).parent.resolve()  # draw/

# Plot settings
FIGURE_SIZE = (12, 8)
DPI = 150
SMOOTHING_WINDOW = 50  # Rolling window for smoothing curves


def load_tensorboard_data(log_dir: Path, tag: str) -> pd.DataFrame:
    """
    Load scalar data from TensorBoard event files.

    Args:
        log_dir: Path to TensorBoard log directory
        tag: Scalar tag to load (e.g., 'Train/mean_reward')

    Returns:
        DataFrame with 'step', 'value', and 'wall_time' columns
    """
    ea = event_accumulator.EventAccumulator(
        str(log_dir),
        size_guidance={
            event_accumulator.SCALARS: 0,  # Load all scalars
        },
    )
    ea.Reload()

    if tag not in ea.Tags()["scalars"]:
        print(f"Warning: Tag '{tag}' not found in {log_dir}")
        print(f"Available tags: {ea.Tags()['scalars']}")
        return pd.DataFrame()

    events = ea.Scalars(tag)

    data = {
        "step": [e.step for e in events],
        "value": [e.value for e in events],
        "wall_time": [e.wall_time for e in events],
    }

    df = pd.DataFrame(data)

    # Convert wall_time to relative time (seconds from start)
    if len(df) > 0:
        df["time"] = df["wall_time"] - df["wall_time"].iloc[0]
        df["time_minutes"] = df["time"] / 60  # Convert to minutes
        df["time_hours"] = df["time"] / 3600  # Convert to hours

    return df


def smooth_curve(values: np.ndarray, window: int = 50) -> np.ndarray:
    """Apply rolling mean smoothing to curve."""
    if len(values) < window:
        return values
    return pd.Series(values).rolling(window=window, min_periods=1).mean().values


def plot_training_reward_vs_time():
    """
    Plot mean reward vs training time for all algorithms.
    """
    print("Loading TensorBoard data...")

    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="deep")
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.labelsize"] = 14
    plt.rcParams["axes.titlesize"] = 16
    plt.rcParams["legend.fontsize"] = 12

    fig, ax = plt.subplots(figsize=FIGURE_SIZE)

    # Color palette
    colors = {
        "PPO": "#2196F3",  # Blue
        "PPG": "#4CAF50",  # Green
        "SAC": "#FF5722",  # Orange
    }

    # Different reward tags for different algorithms
    reward_tags = {
        "PPO": "Train/mean_reward",
        "PPG": "Train/mean_reward",
        "SAC": "Train/mean_episode_reward",
    }

    all_data = {}

    for algo_name, log_dir in RUNS_CONFIG.items():
        if not log_dir.exists():
            print(f"Warning: {log_dir} does not exist, skipping {algo_name}")
            continue

        tag = reward_tags[algo_name]
        df = load_tensorboard_data(log_dir, tag)

        # SAC rewards were scaled by 10x during training, scale back
        if algo_name == "SAC":
            df["value"] = df["value"] / 10.0

        if df.empty:
            print(f"Warning: No data loaded for {algo_name}")
            continue

        all_data[algo_name] = df

        # Smooth the curve
        smoothed_values = smooth_curve(df["value"].values, SMOOTHING_WINDOW)

        # Plot smoothed curve
        ax.plot(
            df["time_minutes"],
            smoothed_values,
            label=algo_name,
            color=colors[algo_name],
            linewidth=2,
            alpha=0.9,
        )

        # Plot raw data as light background
        ax.fill_between(
            df["time_minutes"],
            smooth_curve(df["value"].values, SMOOTHING_WINDOW // 2)
            - df["value"].std() * 0.3,
            smooth_curve(df["value"].values, SMOOTHING_WINDOW // 2)
            + df["value"].std() * 0.3,
            alpha=0.2,
            color=colors[algo_name],
        )

        print(
            f"Loaded {algo_name}: {len(df)} points, "
            f"time range: {df['time_minutes'].min():.1f} - {df['time_minutes'].max():.1f} min, "
            f"reward range: {df['value'].min():.2f} - {df['value'].max():.2f}"
        )

    # Labels and title
    ax.set_xlabel("Training Time (minutes)", fontweight="bold")
    ax.set_ylabel("Mean Episode Reward", fontweight="bold")
    ax.set_title("Training Curves: PPO vs PPG vs SAC", fontweight="bold", pad=15)

    # Legend
    ax.legend(loc="lower right", frameon=True, fancybox=True, shadow=True)

    # Grid
    ax.grid(True, alpha=0.3)

    # Tight layout
    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "training_reward_vs_time.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to: {output_path}")

    # Also save as PDF for high quality
    pdf_path = OUTPUT_DIR / "training_reward_vs_time.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"PDF saved to: {pdf_path}")

    plt.show()

    # Print summary statistics
    print("\n" + "=" * 60)
    print("Summary Statistics")
    print("=" * 60)
    for algo_name, df in all_data.items():
        final_reward = (
            df["value"].iloc[-100:].mean() if len(df) >= 100 else df["value"].mean()
        )
        max_reward = df["value"].max()
        total_time = df["time_minutes"].max()
        print(f"{algo_name}:")
        print(
            f"  - Total training time: {total_time:.1f} minutes ({total_time/60:.2f} hours)"
        )
        print(f"  - Final reward (last 100): {final_reward:.2f}")
        print(f"  - Max reward: {max_reward:.2f}")
        print()

    return all_data


def plot_final_and_max_reward(all_data: dict):
    """
    Plot bar chart comparing final reward, max reward, and training time.
    """
    if not all_data:
        print("No data to plot")
        return

    # Set seaborn style
    sns.set_theme(style="whitegrid", palette="deep")

    # Prepare data
    algorithms = list(all_data.keys())

    final_rewards = []
    max_rewards = []
    training_times = []  # in hours
    time_to_max = []  # time to reach max reward (in hours)

    for algo_name in algorithms:
        df = all_data[algo_name]
        final_reward = (
            df["value"].iloc[-100:].mean() if len(df) >= 100 else df["value"].mean()
        )
        max_reward = df["value"].max()
        total_time = df["time_minutes"].max() / 60  # hours
        max_idx = df["value"].idxmax()
        time_at_max = df.loc[max_idx, "time_minutes"] / 60  # hours

        final_rewards.append(final_reward)
        max_rewards.append(max_reward)
        training_times.append(total_time)
        time_to_max.append(time_at_max)

    # Color palette
    colors = {
        "PPO": "#2196F3",
        "PPG": "#4CAF50",
        "SAC": "#FF5722",
    }
    algo_colors = [colors[a] for a in algorithms]

    # Create figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # ===== Plot 1: Final Reward =====
    ax1 = axes[0, 0]
    bars1 = ax1.bar(
        algorithms, final_rewards, color=algo_colors, edgecolor="black", linewidth=1.2
    )
    ax1.set_ylabel("Mean Reward", fontweight="bold")
    ax1.set_title("Final Reward (Last 100 Steps Average)", fontweight="bold")
    ax1.set_ylim(0, max(final_rewards) * 1.2)
    # Add value labels
    for bar, val in zip(bars1, final_rewards):
        ax1.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # ===== Plot 2: Max Reward =====
    ax2 = axes[0, 1]
    bars2 = ax2.bar(
        algorithms, max_rewards, color=algo_colors, edgecolor="black", linewidth=1.2
    )
    ax2.set_ylabel("Mean Reward", fontweight="bold")
    ax2.set_title("Maximum Reward Achieved", fontweight="bold")
    ax2.set_ylim(0, max(max_rewards) * 1.2)
    # Add value labels
    for bar, val in zip(bars2, max_rewards):
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{val:.2f}",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # ===== Plot 3: Total Training Time =====
    ax3 = axes[1, 0]
    bars3 = ax3.bar(
        algorithms, training_times, color=algo_colors, edgecolor="black", linewidth=1.2
    )
    ax3.set_ylabel("Time (hours)", fontweight="bold")
    ax3.set_title("Total Training Time", fontweight="bold")
    ax3.set_ylim(0, max(training_times) * 1.2)
    # Add value labels
    for bar, val in zip(bars3, training_times):
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}h",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # ===== Plot 4: Time to Max Reward =====
    ax4 = axes[1, 1]
    bars4 = ax4.bar(
        algorithms, time_to_max, color=algo_colors, edgecolor="black", linewidth=1.2
    )
    ax4.set_ylabel("Time (hours)", fontweight="bold")
    ax4.set_title("Time to Reach Maximum Reward", fontweight="bold")
    ax4.set_ylim(0, max(time_to_max) * 1.2)
    # Add value labels
    for bar, val in zip(bars4, time_to_max):
        ax4.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.2,
            f"{val:.1f}h",
            ha="center",
            va="bottom",
            fontweight="bold",
            fontsize=11,
        )

    # Add overall title
    fig.suptitle(
        "Training Performance Comparison: PPO vs PPG vs SAC",
        fontweight="bold",
        fontsize=16,
        y=1.02,
    )

    plt.tight_layout()

    # Save figure
    output_path = OUTPUT_DIR / "training_comparison_bars.png"
    plt.savefig(output_path, dpi=DPI, bbox_inches="tight", facecolor="white")
    print(f"\nBar chart saved to: {output_path}")

    pdf_path = OUTPUT_DIR / "training_comparison_bars.pdf"
    plt.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    print(f"PDF saved to: {pdf_path}")

    plt.show()


def main():
    """Main entry point."""
    print("=" * 60)
    print("Training Results Visualization")
    print("=" * 60)
    print(f"\nConfiguration:")
    for algo, path in RUNS_CONFIG.items():
        exists = "✓" if path.exists() else "✗"
        print(f"  {algo}: {path} [{exists}]")
    print()

    all_data = plot_training_reward_vs_time()
    plot_final_and_max_reward(all_data)


if __name__ == "__main__":
    main()
