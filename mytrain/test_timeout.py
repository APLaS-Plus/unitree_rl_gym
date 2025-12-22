"""
测试超时处理是否正确设置
验证 env.step() 返回的 infos 中是否包含 "time_outs"
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from isaacgym import gymapi
from legged_gym.envs import *
from legged_gym.utils import task_registry

import torch
from argparse import Namespace

# ============================================
# 配置参数（与 sac_legged.py 保持一致）
# ============================================
task = "go2"
num_envs = 16  # 测试用，环境数量小一些
headless = True
sim_device = "cuda:0"
rl_device = "cuda:0"


def main():
    print("=" * 60)
    print("超时处理测试脚本")
    print("=" * 60)
    
    # ============================================
    # 1. 创建环境（与 sac_legged.py 一致）
    # ============================================
    print(f"\n创建环境: task={task}, num_envs={num_envs}")
    
    env_cfg, train_cfg = task_registry.get_cfgs(name=task)
    env_cfg.env.num_envs = num_envs
    
    # 检查 send_timeouts 的默认值
    print(f"\n[检查] env_cfg.env.send_timeouts 原始值: {env_cfg.env.send_timeouts}")
    
    # 强制确保 send_timeouts 为 True
    env_cfg.env.send_timeouts = True
    print(f"[设置] env_cfg.env.send_timeouts = True")
    
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
    
    print(f"\n观测维度: {env.num_obs}")
    print(f"动作维度: {env.num_actions}")
    print(f"环境数量: {env.num_envs}")
    print(f"最大 Episode 长度: {env.max_episode_length} steps")
    
    # ============================================
    # 2. 运行几步，检查 infos 中是否有 time_outs
    # ============================================
    print("\n" + "=" * 60)
    print("开始运行测试...")
    print("=" * 60)
    
    obs = env.get_observations()
    env._resample_commands(torch.arange(num_envs, device=obs.device))
    obs = env.get_observations()
    
    time_outs_found = False
    max_test_steps = int(env.max_episode_length) + 100  # 运行超过一个 episode
    
    for step in range(max_test_steps):
        # 随机动作
        actions = torch.rand(num_envs, env.num_actions, device=obs.device) * 2 - 1
        
        # 执行动作
        next_obs, privileged_obs, rewards, dones, infos = env.step(actions)
        
        # 检查 infos 中是否有 time_outs
        if "time_outs" in infos:
            time_outs = infos["time_outs"]
            num_timeouts = time_outs.sum().item()
            num_dones = dones.sum().item()
            
            if num_timeouts > 0:
                time_outs_found = True
                print(f"\n[Step {step}] ✅ 发现 time_outs!")
                print(f"  - dones 总数: {int(num_dones)}")
                print(f"  - time_outs 总数: {int(num_timeouts)}")
                print(f"  - 真实终止 (摔倒等): {int(num_dones - num_timeouts)}")
                
                # 验证 real_dones 的计算
                real_dones = dones & ~time_outs
                print(f"  - real_dones (dones & ~time_outs): {real_dones.sum().item()}")
                break
        
        obs = next_obs
        
        if step % 500 == 0 and step > 0:
            print(f"[Step {step}] 仍在运行，尚未观察到 time_outs...")
    
    # ============================================
    # 3. 输出结果
    # ============================================
    print("\n" + "=" * 60)
    print("测试结果")
    print("=" * 60)
    
    if time_outs_found:
        print("✅ 测试通过！")
        print("   环境正确返回了 'time_outs' 字段。")
        print("   您可以在 sac_legged.py 中使用以下代码区分超时和真实终止：")
        print()
        print("   real_dones = dones.clone()")
        print("   if 'time_outs' in infos:")
        print("       real_dones = dones & ~infos['time_outs']")
        print()
    else:
        print("❌ 测试失败！")
        print("   在 {} 步内未观察到 'time_outs'。".format(max_test_steps))
        print("   请检查：")
        print("   1. env_cfg.env.send_timeouts 是否被正确设置为 True")
        print("   2. 环境的 max_episode_length 是否过长")


if __name__ == '__main__':
    main()
