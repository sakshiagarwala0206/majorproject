import argparse
import gymnasium
import numpy as np
import torch
import os
import sys
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb
from gymnasium.envs.registration import register
import random

# 📁 Add root for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole_rl import CartPoleEnv
from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import DQN
logger = setup_logger()

# ✅ Register environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole_rl:CartPoleEnv',
)

# 🧾 Config loader
def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

# 📈 Evaluation metrics
def get_convergence_time(rewards, threshold=0.95):
    for episode, reward in enumerate(rewards):
        if reward >= threshold:
            return episode
    return len(rewards)

def get_avg_reward(rewards):
    return np.mean(rewards)

def get_overshoot(angles, target_angle=0):
    max_deviation = max(abs(angle - target_angle) for angle in angles)
    return max_deviation

def get_settling_time(times, angles, target_angle=0, tolerance=0.05):
    for t, angle in zip(times, angles):
        if abs(angle - target_angle) <= tolerance:
            return t
    return max(times)

def get_fall_rate(falls, total_episodes):
    return (falls / total_episodes) * 100

def get_energy(torques):
    return np.sum(np.abs(torques))

def get_smoothness(accelerations, time_steps):
    jerks = np.diff(accelerations) / np.diff(time_steps)
    return np.mean(np.abs(jerks))

def get_robustness(performances):
    return np.std(performances)

# 🧪 Evaluation loop
def evaluate_controller(env, controller, num_episodes, max_steps,epsilon=0.1):
    rewards = []
    overshoots = []
    settling_times = []
    fall_count = 0
    energy = []
    smoothness = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0
        episode_torques = []
        episode_angles = []
        episode_times = []

        for step in range(max_steps):
            # Add noise with epsilon-greedy strategy
            if random.uniform(0, 1) < epsilon:
                # Random action for exploration
                action = env.action_space.sample()  # Random action
            else:
                # Predict the action using the controller (DQN model)
                action = controller.act(obs)
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            total_reward += reward
            episode_times.append(step)
            episode_torques.append(1 if action == 1 else -1)
            episode_angles.append(obs[2])  # pole angle

            if done:
                if abs(obs[2]) > 0.8:
                    fall_count += 1
                break

        rewards.append(total_reward)
        overshoots.append(get_overshoot(episode_angles))
        settling_times.append(get_settling_time(episode_times, episode_angles))
        energy.append(get_energy(episode_torques))
        smoothness.append(get_smoothness(episode_angles, episode_times))

    convergence_time = get_convergence_time(rewards)
    avg_reward = get_avg_reward(rewards)
    fall_rate = get_fall_rate(fall_count, num_episodes)
    robustness = get_robustness(rewards)

    metrics = {
        "Convergence Time (ep)": convergence_time,
        "Avg Reward": avg_reward,
        "Overshoot (°)": np.mean(overshoots),
        "Settling Time (s)": np.mean(settling_times),
        "Fall Rate (%)": fall_rate,
        "Energy (∑|τ|)": np.mean(energy),
        "Smoothness (Jerk)": np.mean(smoothness),
        "Robustness": robustness
    }

    return rewards, metrics, overshoots, settling_times, energy, smoothness

# 🚀 Main testing entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to controller config file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to DQN model (.zip)")
    parser.add_argument('--model_type', type=str, required=True, choices=['DQN', 'DDPG', 'SAC', 'PPO'], help="Model type (DQN, DDPG, SAC, PPO)")
    parser.add_argument('--use_wandb', action='store_true', help="Enable wandb logging")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="assistive-walker-eval",
            config=vars(config),
            name=f"eval_dqn_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # 🎮 Setup env
    env = gymnasium.make("CartPole-v1", render_mode="human")
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    np.random.seed(config.seed)

    # 📦 Load DQN model
    model = DQN.load(args.model_path)
    model_type = args.model_type
    controller = DRLController(model,model_type, action_space=env.action_space)

    logger.info(f"🔍 Evaluating DQN controller...")

    # 🧪 Evaluate
    rewards, metrics, overshoots, settling_times, energy, smoothness = evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
    mean_reward = np.mean(rewards)

    logger.info(f"📊 Mean reward over {config.eval_episodes} episodes: {mean_reward}")

    if args.use_wandb:
        table = wandb.Table(columns=["Episode", "Reward", "Overshoot", "Settling Time", "Energy", "Smoothness"])
        for ep in range(config.eval_episodes):
            table.add_data(ep, rewards[ep], overshoots[ep], settling_times[ep], energy[ep], smoothness[ep])

        wandb.log({
            "mean_reward": mean_reward,
            **metrics,
            "Evaluation Episodes": table
        })
        wandb.finish()

    env.close()
