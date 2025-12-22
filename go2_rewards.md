# GO2 机器人奖励配置指南

本文档详细说明了 GO2 四足机器人强化学习训练中所有可调整的奖励参数。

---

## 📋 当前配置概览

在 `legged_gym/envs/go2/go2_config.py` 中，GO2 场景当前只覆盖了 **2 个奖励权重**：

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9        # 关节限位软阈值（90%）
    base_height_target = 0.25       # 目标身体高度 [m]
    
    class scales(LeggedRobotCfg.rewards.scales):
        torques = -0.0002           # 扭矩惩罚
        dof_pos_limits = -10.0      # 关节限位惩罚
```

其他所有奖励使用基类 `LeggedRobotCfg` 的默认值（见 `legged_gym/envs/base/legged_robot_config.py`）。

---

## 🎯 奖励权重详解 (`rewards.scales`)

### 1️⃣ 运动跟踪奖励

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| **`tracking_lin_vel`** | `1.0` | 奖励机器人跟踪目标线速度命令的准确性 | ⭐ **核心奖励**，增大可提高速度跟踪精度（建议范围：0.5-2.0） |
| **`tracking_ang_vel`** | `0.5` | 奖励机器人跟踪目标角速度命令的准确性 | 增大可提高转向能力（建议范围：0.2-1.0） |

**奖励计算方式**：
```python
tracking_reward = exp(-error² / tracking_sigma²)
```
其中 `tracking_sigma = 0.25`（可调整）

---

### 2️⃣ 姿态稳定性奖励

| 参数 | 默认值 | GO2值 | 说明 | 调整建议 |
|------|--------|-------|------|----------|
| **`lin_vel_z`** | `-2.0` | - | 惩罚 z 方向（垂直）的线速度 | 防止跳跃行为，-1.0 到 -5.0 合理 |
| **`ang_vel_xy`** | `-0.05` | - | 惩罚 pitch 和 roll 方向的角速度 | ⚠️ 可增大到 **-0.1 ~ -0.5** 提高稳定性 |
| **`orientation`** | `0.0` | - | 惩罚身体姿态偏离水平面 | ⚠️ **建议启用** `-1.0 ~ -5.0` 防止倾倒 |
| **`base_height`** | `0.0` | - | 惩罚身体高度偏离目标值 | 配合 `base_height_target=0.25` 使用，建议 `-1.0 ~ -3.0` |

> [!IMPORTANT]
> **防止倾倒问题**：如果机器人倾向于倒地获取奖励，必须启用 `orientation` 和 `base_height` 惩罚！

---

### 3️⃣ 能量效率与平滑性

| 参数 | 默认值 | GO2值 | 说明 | 调整建议 |
|------|--------|-------|------|----------|
| **`torques`** | `-0.00001` | **`-0.0002`** | 惩罚关节扭矩大小 | 鼓励节能，范围：-0.0001 ~ -0.001 |
| **`dof_vel`** | `0.0` | - | 惩罚关节速度 | 设置 -0.001 ~ -0.01 减少关节快速运动 |
| **`dof_acc`** | `-2.5e-7` | - | 惩罚关节加速度 | 增大到 -1e-6 ~ -1e-5 减少抖动 |
| **`action_rate`** | `-0.01` | - | 惩罚连续动作之间的变化率 | ⚠️ **重要**，增大到 **-0.05 ~ -0.2** 提高平滑度 |

> [!TIP]
> `action_rate` 是控制运动平滑性的关键参数，对于真实机器人部署尤为重要！

---

### 4️⃣ 步态与接触奖励

| 参数 | 默认值 | 说明 | 调整建议 |
|------|--------|------|----------|
| **`feet_air_time`** | `1.0` | 奖励脚离地的时间 | ⭐ 鼓励 trot 步态，保持 0.5-2.0 |
| **`collision`** | `-1.0` | 惩罚非脚部身体部位（如大腿、小腿）接触地面 | 防止膝盖触地，-0.5 ~ -2.0 |
| **`feet_stumble`** | `0.0` | 惩罚脚部在摆动相时的接触 | 建议启用 **-0.1 ~ -1.0** 防止拖脚 |

**相关配置**：
```python
class asset:
    penalize_contacts_on = ["thigh", "calf"]  # collision 惩罚的身体部位
    terminate_after_contacts_on = ["base"]     # 触发终止的身体部位
```

---

### 5️⃣ 安全与约束

| 参数 | 默认值 | GO2值 | 说明 | 调整建议 |
|------|--------|-------|------|----------|
| **`termination`** | `0.0` | - | 提前终止（摔倒）的惩罚 | 设置 **-1.0 ~ -5.0** 强烈避免摔倒 |
| **`dof_pos_limits`** | `0.0` | **`-10.0`** | 惩罚关节位置超出软限制 | GO2 已设置，-5.0 ~ -20.0 合理 |
| **`stand_still`** | `0.0` | - | 当速度命令为 0 时的特殊奖励 | 特殊任务可用，通常保持 0 |

---

## ⚙️ 奖励函数参数

除了权重 `scales` 外，还有以下全局参数可调整：

```python
class rewards:
    # 奖励裁剪
    only_positive_rewards = True          # 是否将负的总奖励裁剪为 0
    
    # 跟踪奖励参数
    tracking_sigma = 0.25                 # 跟踪奖励的高斯标准差
    
    # 软限制阈值（百分比）
    soft_dof_pos_limit = 0.9             # GO2: 0.9 = 90% URDF 限位
    soft_dof_vel_limit = 1.0             # 速度限制百分比
    soft_torque_limit = 1.0              # 扭矩限制百分比
    
    # 目标值
    base_height_target = 0.25            # GO2: 0.25m（站立高度）
    max_contact_force = 100.             # 接触力惩罚阈值 [N]
```

### 参数说明

| 参数 | 典型值 | 说明 |
|------|--------|------|
| `only_positive_rewards` | `True` | 防止早期训练中因大量负奖励导致的提前终止问题 |
| `tracking_sigma` | 0.15-0.3 | 越小对跟踪误差越敏感，奖励曲线越陡峭 |
| `soft_dof_pos_limit` | 0.85-0.95 | 低于 URDF 限位的安全边界，防止碰到硬限制 |
| `base_height_target` | 0.2-0.3 | GO2 机器人合理站立高度约 0.25m |
| `max_contact_force` | 50-200 | 超过此值的接触力会被惩罚 |

---

## 📊 推荐配置方案

### 🟢 方案 1：稳定性优先（防止倾倒）

适用于早期训练或遇到机器人倾倒问题时：

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    tracking_sigma = 0.25
    
    class scales(LeggedRobotCfg.rewards.scales):
        # 跟踪
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        
        # 稳定性（加强）
        orientation = -5.0          # 🔥 强烈惩罚倾倒
        base_height = -2.0          # 🔥 保持站立高度
        lin_vel_z = -2.0
        ang_vel_xy = -0.2           # 🔥 增强姿态稳定
        
        # 安全
        termination = -2.0          # 🔥 惩罚摔倒
        dof_pos_limits = -10.0
        collision = -1.0
        feet_stumble = -0.5         # 🔥 防止拖脚
        
        # 平滑性
        action_rate = -0.05
        dof_acc = -5e-7
        
        # 能量
        torques = -0.0002
        
        # 步态
        feet_air_time = 1.0
```

---

### 🟡 方案 2：性能优先（已稳定行走）

适用于机器人已能稳定行走，需要优化速度和效率时：

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    tracking_sigma = 0.2            # 🔥 更严格的跟踪
    
    class scales(LeggedRobotCfg.rewards.scales):
        # 跟踪（加强）
        tracking_lin_vel = 1.5      # 🔥 提高速度跟踪权重
        tracking_ang_vel = 0.8      # 🔥 提高转向能力
        
        # 稳定性（适中）
        orientation = -2.0
        base_height = -1.0
        lin_vel_z = -2.0
        ang_vel_xy = -0.1
        
        # 平滑性（加强）
        action_rate = -0.1          # 🔥 更平滑的动作
        dof_vel = -0.005            # 🔥 限制关节速度
        dof_acc = -1e-6             # 🔥 减少抖动
        
        # 能量效率（加强）
        torques = -0.0005           # 🔥 鼓励节能
        
        # 步态（加强）
        feet_air_time = 1.5         # 🔥 更明显的步态
        
        # 安全
        termination = -1.0
        dof_pos_limits = -10.0
        collision = -1.0
```

---

### 🔵 方案 3：平衡配置（通用推荐）

适用于大多数场景的平衡配置：

```python
class rewards(LeggedRobotCfg.rewards):
    soft_dof_pos_limit = 0.9
    base_height_target = 0.25
    tracking_sigma = 0.25
    
    class scales(LeggedRobotCfg.rewards.scales):
        # 跟踪
        tracking_lin_vel = 1.0
        tracking_ang_vel = 0.5
        
        # 稳定性
        orientation = -3.0
        base_height = -1.5
        lin_vel_z = -2.0
        ang_vel_xy = -0.15
        
        # 平滑性
        action_rate = -0.08
        dof_acc = -5e-7
        
        # 能量
        torques = -0.0002
        dof_vel = -0.002
        
        # 步态
        feet_air_time = 1.0
        
        # 安全
        termination = -1.5
        dof_pos_limits = -10.0
        collision = -1.0
        feet_stumble = -0.3
```

---

## 🔧 调试技巧

### 1. 监控奖励分布

在训练时使用 TensorBoard 监控各项奖励的贡献：

```bash
tensorboard --logdir=logs/
```

关注以下指标：
- `rewards/tracking_lin_vel` - 应该是主要正奖励
- `rewards/feet_air_time` - 步态相关正奖励
- `rewards/orientation` - 不应该过大导致机器人过度保守
- `Episode/rew_total` - 总奖励趋势

---

### 2. 调整原则

> [!WARNING]
> **数量级匹配**：确保各奖励项在训练中的实际贡献值处于同一数量级（通常 0.1-10），避免某项主导整个奖励函数。

**调整步骤**：
1. **观察 TensorBoard**：查看各项奖励的实际数值
2. **识别主导项**：如果某项奖励贡献远大于其他（>10倍），考虑降低其权重
3. **逐步调整**：每次只改变 1-2 个权重，观察 10-50 iterations 效果
4. **验证行为**：在仿真环境中可视化机器人行为是否符合预期

---

### 3. 常见问题诊断

| 问题现象 | 可能原因 | 解决方案 |
|---------|---------|---------|
| 机器人倾倒获取奖励 | `orientation` 未启用 | 设置 `orientation = -3.0 ~ -5.0` |
| 动作抖动严重 | `action_rate` 太小 | 增大到 `-0.1` 或更大 |
| 不跟踪速度命令 | `tracking_lin_vel` 权重不足 | 增大到 `1.5` 或调小 `tracking_sigma` |
| 膝盖触地 | `collision` 权重不足 | 增大到 `-2.0` 或调整 `penalize_contacts_on` |
| 步态不明显 | `feet_air_time` 太小 | 增大到 `1.5 ~ 2.0` |
| 关节超限 | `dof_pos_limits` 太小 | 增大到 `-15.0 ~ -20.0` |

---

## 📚 相关文件

- **GO2 配置**：`legged_gym/envs/go2/go2_config.py`
- **基类配置**：`legged_gym/envs/base/legged_robot_config.py`
- **奖励计算逻辑**：`legged_gym/envs/base/legged_robot.py` 中的 `_reward_*()` 方法

---

## 🚀 快速开始

1. 复制上述推荐配置方案到 `go2_config.py`
2. 根据训练表现微调权重
3. 使用 TensorBoard 监控奖励分布
4. 迭代优化直到达到预期性能

> [!TIP]
> 建议使用 Git 版本控制记录每次奖励调整，方便回溯和对比不同配置的效果！

---

**最后更新**：2025-12-18  
**适用版本**：unitree_rl_gym
