"""
MuJoCo Sim2Sim Deployment Script for Go2 Robot

This script loads a trained policy and runs it in MuJoCo simulator for sim2sim validation.
The observation structure is adapted for Go2 (48 dims, with base_lin_vel, no phase signals).

Observation Structure (48 dims):
    [0:3]   base_lin_vel * lin_vel_scale
    [3:6]   base_ang_vel * ang_vel_scale
    [6:9]   projected_gravity
    [9:12]  commands * cmd_scale
    [12:24] (dof_pos - default_angles) * dof_pos_scale
    [24:36] dof_vel * dof_vel_scale
    [36:48] previous_actions

Usage:
    python mytrain/deploy/deploy_go2_mujoco.py go2.yaml
    python mytrain/deploy/deploy_go2_mujoco.py go2.yaml --cmd 0.5,0,0.5

Requirements:
    - mujoco >= 2.3.0
    - PyYAML
    - torch
"""

import os
import sys
import time
import argparse
from pathlib import Path

import numpy as np
import yaml

# Add paths
SCRIPT_DIR = Path(__file__).parent.resolve()
MYTRAIN_DIR = SCRIPT_DIR.parent.resolve()
ROOT_DIR = MYTRAIN_DIR.parent.resolve()

sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MYTRAIN_DIR))

# Import after path setup
try:
    import mujoco
    import mujoco.viewer

    MUJOCO_AVAILABLE = True
except ImportError:
    MUJOCO_AVAILABLE = False
    print("Warning: MuJoCo not available. Install with: pip install mujoco")

import torch


def get_gravity_orientation(quaternion: np.ndarray) -> np.ndarray:
    """
    Compute the gravity vector in body frame from quaternion.

    The quaternion format is [w, x, y, z] in MuJoCo.
    Returns projected gravity in body frame.
    """
    qw, qx, qy, qz = quaternion

    # Rotate gravity vector [0, 0, -1] by inverse of body quaternion
    # This gives gravity direction in body frame
    gravity = np.zeros(3)
    gravity[0] = 2 * (-qz * qx + qw * qy)
    gravity[1] = -2 * (qz * qy + qw * qx)
    gravity[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity


def pd_control(
    target_q: np.ndarray,
    q: np.ndarray,
    kp: np.ndarray,
    target_dq: np.ndarray,
    dq: np.ndarray,
    kd: np.ndarray,
) -> np.ndarray:
    """
    Compute PD control torques.

    tau = kp * (target_q - q) + kd * (target_dq - dq)
    """
    return kp * (target_q - q) + kd * (target_dq - dq)


def estimate_base_velocity(pos_history: list, dt: float) -> np.ndarray:
    """
    Estimate base linear velocity using finite difference.

    Args:
        pos_history: List of recent base positions [(x, y, z), ...]
        dt: Time step between positions

    Returns:
        Estimated velocity in world frame
    """
    if len(pos_history) < 2:
        return np.zeros(3)

    vel = (np.array(pos_history[-1]) - np.array(pos_history[-2])) / dt
    return vel


def world_to_body_velocity(world_vel: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """
    Transform velocity from world frame to body frame.

    Args:
        world_vel: Velocity in world frame
        quaternion: Body orientation quaternion [w, x, y, z]

    Returns:
        Velocity in body frame
    """
    qw, qx, qy, qz = quaternion

    # Rotation matrix from world to body (transpose of body to world)
    # This is the inverse rotation
    R = np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy + qw * qz),
                2 * (qx * qz - qw * qy),
            ],
            [
                2 * (qx * qy - qw * qz),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz + qw * qx),
            ],
            [
                2 * (qx * qz + qw * qy),
                2 * (qy * qz - qw * qx),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ]
    )

    # Transform to body frame (inverse rotation)
    body_vel = R.T @ world_vel
    return body_vel


def main():
    parser = argparse.ArgumentParser(description="Go2 MuJoCo Deployment")
    parser.add_argument(
        "config_file",
        type=str,
        help="Config file name in the deploy folder (e.g., go2.yaml)",
    )
    parser.add_argument(
        "--cmd",
        type=str,
        default=None,
        help="Override velocity command: vx,vy,vyaw (e.g., 1.0,0,0)",
    )
    args = parser.parse_args()

    if not MUJOCO_AVAILABLE:
        print("Error: MuJoCo is required for this script.")
        print("Install with: pip install mujoco")
        return

    # Load config
    config_path = SCRIPT_DIR / args.config_file
    if not config_path.exists():
        config_path = SCRIPT_DIR / "configs" / args.config_file

    if not config_path.exists():
        print(f"Error: Config file not found: {args.config_file}")
        print(f"Searched: {SCRIPT_DIR}")
        return

    print(f"Loading config: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Parse paths
    policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT_DIR))
    xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", str(ROOT_DIR))

    # Check files exist
    if not Path(policy_path).exists():
        print(f"Error: Policy file not found: {policy_path}")
        print("Run export_jit.py first to export the policy.")
        return

    if not Path(xml_path).exists():
        print(f"Error: MuJoCo XML not found: {xml_path}")
        print("Download Go2 model from MuJoCo Menagerie.")
        return

    # Load parameters
    simulation_duration = config["simulation_duration"]
    simulation_dt = config["simulation_dt"]
    control_decimation = config["control_decimation"]
    control_dt = simulation_dt * control_decimation

    kps = np.array(config["kps"], dtype=np.float32)
    kds = np.array(config["kds"], dtype=np.float32)
    default_angles = np.array(config["default_angles"], dtype=np.float32)

    lin_vel_scale = config["lin_vel_scale"]
    ang_vel_scale = config["ang_vel_scale"]
    dof_pos_scale = config["dof_pos_scale"]
    dof_vel_scale = config["dof_vel_scale"]
    action_scale = config["action_scale"]
    cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

    num_actions = config["num_actions"]
    num_obs = config["num_obs"]

    # Velocity command
    if args.cmd:
        cmd = np.array([float(x) for x in args.cmd.split(",")], dtype=np.float32)
    else:
        cmd = np.array(config["cmd_init"], dtype=np.float32)

    print(f"\n{'='*60}")
    print("Go2 MuJoCo Deployment")
    print(f"{'='*60}")
    print(f"Policy: {policy_path}")
    print(f"MuJoCo XML: {xml_path}")
    print(f"Control frequency: {1/control_dt:.1f} Hz")
    print(f"Velocity command: vx={cmd[0]:.2f}, vy={cmd[1]:.2f}, vyaw={cmd[2]:.2f}")
    print(f"{'='*60}\n")

    # Initialize state variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)
    pos_history = []  # For velocity estimation

    # Load MuJoCo model
    print("Loading MuJoCo model...")
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # Load policy
    print("Loading policy...")
    policy = torch.jit.load(policy_path)
    policy.eval()

    counter = 0

    print("Starting simulation...")

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()

        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()

            # Apply PD control
            tau = pd_control(
                target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds
            )
            d.ctrl[:] = tau

            # Step physics
            mujoco.mj_step(m, d)

            # Track base position for velocity estimation
            pos_history.append(d.qpos[:3].copy())
            if len(pos_history) > 10:
                pos_history.pop(0)

            counter += 1

            # Policy update at control frequency
            if counter % control_decimation == 0:
                # Get sensor readings
                qj = d.qpos[7:]  # Joint positions
                dqj = d.qvel[6:]  # Joint velocities
                quat = d.qpos[3:7]  # Base orientation [w, x, y, z]
                omega = d.qvel[3:6]  # Base angular velocity

                # Estimate base linear velocity
                world_vel = estimate_base_velocity(pos_history, simulation_dt)
                body_vel = world_to_body_velocity(world_vel, quat)

                # Compute gravity orientation
                gravity = get_gravity_orientation(quat)

                # Build observation (48 dims for Go2)
                obs[:3] = body_vel * lin_vel_scale
                obs[3:6] = omega * ang_vel_scale
                obs[6:9] = gravity
                obs[9:12] = cmd * cmd_scale
                obs[12:24] = (qj - default_angles) * dof_pos_scale
                obs[24:36] = dqj * dof_vel_scale
                obs[36:48] = action

                # Policy inference
                obs_tensor = torch.from_numpy(obs).unsqueeze(0).float()
                with torch.no_grad():
                    action = policy(obs_tensor).numpy().squeeze()

                # Convert action to target joint positions
                target_dof_pos = action * action_scale + default_angles

            # Sync viewer
            viewer.sync()

            # Timing
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    print(f"\nSimulation completed. Duration: {time.time() - start:.1f}s")


if __name__ == "__main__":
    main()
