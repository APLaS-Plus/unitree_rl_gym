import os
from pathlib import Path
import datetime

import torch
import torch.nn as nn
import torch.functional as f
from torch.utils.tensorboard import SummaryWriter
from torchrl.collectors import SyncDataCollector
from torchrl.data import LazyMemmapStorage, ReplayBuffer, TensorDictReplayBuffer
from torchrl.data.replay_buffers.storages import ListStorage
from torchrl.data.replay_buffers.samplers import RandomSampler
from tensordict import TensorDict
import tensorboard

import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from rich.progress import track
from rich.traceback import install
install(show_locals=True)

from models import *

# config
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

gamma = 0.99
episodes = 1_000_000
save_model_freq = 10000
tau = 0.001
# target_action_noise_std = 0.2
# target_action_noise_clip = 0.5
# policy_delay_freq = 2
target_update_freq = 1000
actor_lr = 5e-5
critic_lr = 5e-4
start_rb_len = 10_000
rb_size = start_rb_len * 10
bg_noise_std = 0.5
ed_noise_std = 0.05
noise_decay_steps = episodes // 3
batch_size = 128
eval_freq = 10000
rms_update_freq = 10000
rms_update_batch_size = 5000


ROOT_PATH = Path(__file__).parent.resolve()
runs_base_path = ROOT_PATH / "runs"
os.makedirs(str(runs_base_path), exist_ok=True)

# Find next available run number
run_num = 1
while (runs_base_path / f"run{run_num}").exists():
    run_num += 1

model_save_path = runs_base_path / f"run{run_num}"
tensorboard_path = runs_base_path / "tensorboard" / f"run{run_num}"
os.makedirs(str(model_save_path), exist_ok=True)
os.makedirs(str(tensorboard_path), exist_ok=True)

def linear_decrease(bg, ed, len, i):
    return ed + max(len - i, 0)/len * bg

def soft_update(target, online, tau):
    for target_param, online_param in zip(target.parameters(), online.parameters()):
        target_param.data.copy_(tau * online_param.data + (1.0 - tau) * target_param.data)

# make env and test
print("build env")
env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,  enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
test_env = gym.make("LunarLander-v3", continuous=True, gravity=-10.0,  enable_wind=True, wind_power=15.0, turbulence_power=1.5, render_mode="rgb_array")
observation, info = env.reset()

action_sample = env.action_space.sample()
action_dim = env.action_space.shape[0]
action_high = torch.tensor(env.action_space.high, dtype=torch.float32, device=device)
action_low = torch.tensor(env.action_space.low, dtype=torch.float32, device=device)
print(f"action sample {action_sample}")
print(f"action dim {action_dim}, high {action_high}, low {action_low}")

observation, reward, terminated, truncated, _ = env.step(action_sample)
print(f"observation {observation}")
print(f"observation dim {len(observation)}")
print(f"reward {reward}")
print(f"terminated {terminated}")
print(f"truncated {truncated}")

# model define
actor = StochasticActor(len(observation), action_dim, action_high).to(device)
critic1 = Critic(len(observation), action_dim).to(device)
critic2 = Critic(len(observation), action_dim).to(device)

target_actor = StochasticActor(len(observation), action_dim, action_high).to(device)
target_critic1 = Critic(len(observation), action_dim).to(device)
target_critic2 = Critic(len(observation), action_dim).to(device)

target_actor.load_state_dict(actor.state_dict())
target_critic1.load_state_dict(critic1.state_dict())
target_critic2.load_state_dict(critic2.state_dict())

target_actor.eval()
target_critic1.eval()
target_critic2.eval()

# rms = RuningMeanStd(len(observation)).to(device)

log_alpha = torch.zeros(1, requires_grad=True, device=device)
alpha = log_alpha.exp()

torch.compile(actor)
torch.compile(critic1)
torch.compile(critic2)
torch.compile(target_actor)
torch.compile(target_critic1)
torch.compile(target_critic2)
# torch.compile(rms)

actor_opti = torch.optim.Adam(actor.parameters(), lr=actor_lr)
critic_opti = torch.optim.Adam(list(critic1.parameters()) + list(critic2.parameters()), lr=critic_lr)
alpha_opti = torch.optim.Adam([log_alpha], lr=actor_lr)

lossfn = nn.MSELoss()



writer = SummaryWriter(log_dir=str(tensorboard_path))

# rb
rb = TensorDictReplayBuffer(
    storage=ListStorage(max_size=rb_size),
    sampler=RandomSampler(),
    batch_size=batch_size,
    # device=device,
)

target_entropy = -torch.prod(torch.tensor(env.action_space.shape, dtype=torch.float32)).item()
print(f"Target Entropy: {target_entropy}")

# train
obs, _ = env.reset()
best_reward = float('-inf')
for i in track(range(episodes + start_rb_len), description="Train"):
    if i <= start_rb_len:
        action = env.action_space.sample()
    else:
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
            # norm_obs = rms(obs_tensor)
            # action_tensor, _ = actor.sample(norm_obs)
            action_tensor, _ = actor.sample(obs_tensor)
            action = action_tensor.cpu().numpy().flatten()

    data = {
        "obs": torch.tensor(obs, dtype=torch.float32),
        "action": torch.tensor(action, dtype=torch.float32),
    }

    obs, rw, terminated, truncated, _ = env.step(action)

    data[("next", "obs")] = torch.tensor(obs, dtype=torch.float32)
    data[("next", "rw")] = torch.tensor([rw], dtype=torch.float32)
    data[("next", "done")] = torch.tensor([terminated or truncated], dtype=torch.bool)

    data = TensorDict(data, batch_size=[])
    rb.add(data.to(device))
    
    if terminated or truncated:
        obs, _ = env.reset()

    if i >= start_rb_len:
        # if i % rms_update_freq == 0:
        #     obs_batch = rb.sample(batch_size=rms_update_batch_size).to(device).get("obs")
            # _ = rms(obs_batch, True)
        simple = rb.sample().to(device)
        obs_batch = simple.get("obs")
        action_batch = simple.get("action")
        next_obs_batch = simple.get(("next", "obs"))
        reward_batch = simple.get(("next", "rw"))
        done_batch = simple.get(("next", "done"))

        # obs_batch = rms(obs_batch)
        # next_obs_batch = rms(next_obs_batch)
        
        with torch.no_grad():
            # add noise to next action
            next_actions, next_log_prob = actor.sample(next_obs_batch)


            # choose the min next q target
            next_q_target1 = target_critic1(next_obs_batch, next_actions)
            next_q_target2 = target_critic2(next_obs_batch, next_actions)

            next_q_target = torch.min(next_q_target1, next_q_target2)

            alpha = log_alpha.exp().detach()
            next_q_value = next_q_target - (alpha * next_log_prob)
            
            target_q = reward_batch + (gamma * next_q_value * (~done_batch).float())

        curr_q1 = critic1(obs_batch, action_batch)
        curr_q2 = critic2(obs_batch, action_batch)

        critic1_loss = lossfn(curr_q1, target_q)
        critic2_loss = lossfn(curr_q2, target_q)

        critic_loss = critic1_loss + critic2_loss

        # update critic
        critic_opti.zero_grad()
        critic_loss.backward()
        critic_opti.step()

        # update actor

        actions_pi, log_prob_pi = actor.sample(obs_batch)
        
        q1_pi = critic1(obs_batch, actions_pi)
        q2_pi = critic2(obs_batch, actions_pi)
        
        min_q_pi = torch.min(q1_pi, q2_pi)

        alpha = log_alpha.exp().detach()
        actor_loss = (alpha * log_prob_pi - min_q_pi).mean()

        actor_opti.zero_grad()
        actor_loss.backward()
        actor_opti.step()

        alpha_loss = -(log_alpha * (log_prob_pi + target_entropy).detach()).mean()
        
        alpha_opti.zero_grad()
        alpha_loss.backward()
        alpha_opti.step()
        
        soft_update(target_actor, actor, tau)
        soft_update(target_critic1, critic1, tau)
        soft_update(target_critic2, critic2, tau)

        writer.add_scalar("train/sac_critic1_loss", critic1_loss, i)
        writer.add_scalar("train/sac_critic2_loss", critic2_loss, i)
        writer.add_scalar("train/sac_critic_loss", critic_loss, i)
        writer.add_scalar("train/sac_actor_loss", actor_loss, i)
        writer.add_scalar("train/sac_alpha_loss", alpha_loss, i)
        # writer.add_scalar("train/sac_noise_std", noise_std, i)
        writer.add_scalar("train/sac_actor_lr", actor_opti.param_groups[0]['lr'], i)
        writer.add_scalar("train/sac_critic_lr", critic_opti.param_groups[0]['lr'], i)
        writer.add_scalar("train/sac_alpha_lr", alpha_opti.param_groups[0]['lr'], i)

        if i % eval_freq == 0:
            test_rewards = []
            test_episode_counts = []
            for _ in range(10):
                test_obs, _ = test_env.reset()
                test_done = False
                test_sum_reward = 0.0
                test_episode_count = 0
                while(not test_done):
                    test_episode_count += 1
                    # test_obs = rms(test_obs, should_update=False).to(device).cpu().numpy()
                    test_action = actor.get_action(test_obs)
                    test_obs, test_rw, test_terminated, test_truncated, _ = test_env.step(test_action)
                    test_sum_reward += test_rw
                    test_done = test_terminated or test_truncated
                    
                test_episode_counts.append(test_episode_count)
                test_rewards.append(test_sum_reward)

            if np.mean(test_rewards) > best_reward:
                actor.save_state_dict(model_save_path / "best_sac__actor.pth")
                critic1.save_state_dict(model_save_path / "best_sac_critic1.pth")
                critic2.save_state_dict(model_save_path / "best_sac_critic2.pth")
                # rms.save_state_dict(model_save_path / "best_sac_rms.pth")

            writer.add_scalar("eval/reward", np.mean(test_rewards), i//eval_freq)
            writer.add_scalar("eval/len", np.mean(test_episode_counts), i//eval_freq)
            print(f"[info] test {i//eval_freq} use {np.mean(test_episode_counts).item()} steps and get {np.mean(test_rewards).item()} score")

        if i % save_model_freq == 0:
            actor.save_state_dict(model_save_path / "model" / f"sac_actor{i}.pth")
            critic1.save_state_dict(model_save_path / "model" / f"sac_critic1_{i}.pth")
            critic2.save_state_dict(model_save_path / "model" / f"sac_critic2_{i}.pth")
            # rms.save_state_dict(model_save_path / "model" / f"sac_rms{i}.pth")

actor.save_state_dict(model_save_path / "final_sac_actor.pth")
critic1.save_state_dict(model_save_path / "final_sac_critic1.pth")
critic2.save_state_dict(model_save_path / "final_sac_critic2.pth")
# rms.save_state_dict(model_save_path / "final_sac_rms.pth")

writer.flush()
writer.close()
