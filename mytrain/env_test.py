"""
简单的环境测试脚本 - 演示如何调用 legged_gym 的 RL 环境
不包含算法部分，只展示环境的基本使用方式
"""

import os
import sys

# 添加根目录到路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymapi
from isaacgym import gymutil
from legged_gym.envs import *
from legged_gym.utils import task_registry

import torch
import numpy as np
from argparse import Namespace


def main():
    """
    环境基本使用流程演示
    """
    # ============================================
    # 1. 配置参数（直接在脚本里设置）
    # ============================================
    task = "go2"           # 机器人类型: go2, g1, h1, h1_2
    num_envs = 4096        # 并行环境数
    headless = True        # 无头模式
    sim_device = "cuda:0"  # 仿真设备
    rl_device = "cuda:0"   # RL设备
    
    print(f"配置: task={task}, num_envs={num_envs}, headless={headless}")
    print(f"设备: sim_device={sim_device}, rl_device={rl_device}")
    
    # ============================================
    # 2. 获取环境配置
    # ============================================
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    
    # 修改配置
    print(f"原始环境数: {env_cfg.env.num_envs}")
    env_cfg.env.num_envs = num_envs
    print(f"实际环境数: {env_cfg.env.num_envs}")
    
    # ============================================
    # 3. 创建环境
    # ============================================
    # 构建一个简单的 args 对象用于 make_env
    
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
        device="cuda" if "cuda" in sim_device else "cpu",
        compute_device_id=0,
        sim_device_type="cuda",
        sim_device_id=0
    )
    
    env, env_cfg = task_registry.make_env(name=task, args=args, env_cfg=env_cfg)
    
    print("\n========== 环境信息 ==========")
    print(f"观测维度: {env.num_obs}")
    print(f"动作维度: {env.num_actions}")
    print(f"环境数量: {env.num_envs}")
    print(f"最大步数: {env.max_episode_length}")
    print(f"计算设备: {env.device}")
    
    # ============================================
    # 4. 获取初始观测
    # ============================================
    obs = env.get_observations()
    print(f"\n初始观测 shape: {obs.shape}")
    print(f"初始观测类型: {obs.dtype}")
    print(f"初始观测范围: [{obs.min():.4f}, {obs.max():.4f}]")
    
    # ============================================
    # 5. 运行一个 episode
    # ============================================
    print("\n========== 开始运行 episode ==========")
    
    num_steps = 100  # 运行 100 步
    episode_rewards = torch.zeros(env.num_envs, device=env.device)
    episode_lengths = torch.zeros(env.num_envs, device=env.device)
    
    for step in range(num_steps):
        # 生成随机动作 (示例 - 实际应该由你的模型生成)
        actions = torch.randn(env.num_envs, env.num_actions, device=env.device)
        actions = torch.clamp(actions, -1.0, 1.0)  # 限制在[-1, 1]范围
        
        # 执行一步环境
        obs, privileged_obs, rewards, dones, infos = env.step(actions)
        
        # 统计
        episode_rewards += rewards
        episode_lengths += 1
        
        # 打印信息
        if step % 20 == 0:
            print(f"Step {step:3d} | "
                  f"奖励 mean={rewards.mean():.4f} std={rewards.std():.4f} | "
                  f"观测 mean={obs.mean():.4f} std={obs.std():.4f}")
        
        # 检查返回值的形状
        if step == 0:
            print(f"\n返回值 shape:")
            print(f"  obs:            {obs.shape}")
            if privileged_obs is not None:
                print(f"  privileged_obs: {privileged_obs.shape}")
            print(f"  rewards:        {rewards.shape}")
            print(f"  dones:          {dones.shape}")
            print(f"  infos keys:     {list(infos.keys())}")
        
        # 检查是否有环境重置信息
        # if 'episode' in infos:
        #     print(f"  Episode info: {infos['episode']}")
    
    # ============================================
    # 6. 统计结果
    # ============================================
    print("\n========== Episode 统计 ==========")
    print(f"平均奖励: {episode_rewards.mean():.4f}")
    print(f"奖励 std:  {episode_rewards.std():.4f}")
    print(f"平均步数: {episode_lengths.mean():.1f}")
    
    # ============================================
    # 7. 观测重置功能
    # ============================================
    print("\n========== 测试重置功能 ==========")
    
    # 重置所有环境
    reset_ids = torch.arange(min(4, env.num_envs), device=env.device)
    print(f"重置前 episode_length: {env.episode_length_buf[reset_ids]}")
    
    env.reset_idx(reset_ids)
    
    print(f"重置后 episode_length: {env.episode_length_buf[reset_ids]}")
    
    # 再运行几步
    for _ in range(5):
        actions = torch.zeros(env.num_envs, env.num_actions, device=env.device)
        obs, _, _, _, _ = env.step(actions)
    
    print(f"再运行5步后 episode_length: {env.episode_length_buf[reset_ids]}")
    
    # ============================================
    # 8. 访问配置信息
    # ============================================
    print("\n========== 配置信息示例 ==========")
    print(f"重力: {env_cfg.sim.gravity}")
    print(f"仿真频率: {1/env_cfg.sim.dt} Hz")
    print(f"控制频率: {(1/env_cfg.sim.dt) / env_cfg.control.decimation} Hz")
    print(f"默认关节角度数: {len(env_cfg.init_state.default_joint_angles)}")
    
    print("\n✓ 环境测试完成!")


if __name__ == '__main__':
    main()
