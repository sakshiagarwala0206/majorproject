import argparse
import gymnasium
import numpy as np
import torch
import os
import sys
import pickle
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb
from gymnasium.envs.registration import register

# 📁 Add root for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from trash_test_code.cartpole_rl import CartPoleEnv  # Custom env
# 🎮 Controller imports
from controllers.rl_controller import RLAgent
from controllers.drl_controller import DRLController
from controllers.pid_controller import PIDController

# 🧾 Logger
from train.utils.logger import setup_logger
logger = setup_logger()

#
import yaml
from types import SimpleNamespace

# Assuming load_config is a function that loads the YAML configuration
def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)


# ✅ Register your environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole_rl:CartPoleEnv',  # Correct the entry point
)


import numpy as np

# Function to calculate Convergence Time (ep)
def get_convergence_time(rewards, threshold=0.95):
    for episode, reward in enumerate(rewards):
        if reward >= threshold:
            return episode
    return len(rewards)  # if convergence doesn't happen, return max

# Function to calculate Average Reward
def get_avg_reward(rewards):
    return np.mean(rewards)

# Function to calculate Overshoot (in degrees)
def get_overshoot(angles, target_angle=0):
    max_deviation = max(abs(angle - target_angle) for angle in angles)
    return max_deviation

# Function to calculate Settling Time (in seconds)
def get_settling_time(times, angles, target_angle=0, tolerance=0.05):
    for t, angle in zip(times, angles):
        if abs(angle - target_angle) <= tolerance:
            return t
    return max(times)

# Function to calculate Fall Rate (%)
def get_fall_rate(falls, total_episodes):
    return (falls / total_episodes) * 100

# Function to calculate Energy (Sum of Absolute Torques)
def get_energy(torques):
    return np.sum(np.abs(torques))

# Function to calculate Smoothness (Jerk)
def get_smoothness(accelerations, time_steps):
    jerks = np.diff(accelerations) / np.diff(time_steps)  # Jerk = change in acceleration / time
    return np.mean(np.abs(jerks))

# Function to calculate Robustness (Standard Deviation of performance)
def get_robustness(performances):
    return np.std(performances)  # Standard deviation of performance across episodes

def evaluate_controller(env, controller, num_episodes, max_steps, controller_type="qlearning"):
    rewards = []
    overshoots = []
    settling_times = []
    fall_count = 0
    energy = []
    smoothness = []
    performances = []

    for episode in range(num_episodes):
        obs, _ = env.reset(seed=episode)
        total_reward = 0
        episode_torques = []
        episode_angles = []
        episode_times = []

        for step in range(max_steps):
            action = controller.act(obs)

            # If controller is PID, action is torque
            # If controller is Q-learning or DQN, action is discrete
            if controller_type in ["qlearning", "dqn"]:
                real_action = action  # Discrete action (0 or 1)
            else:  # pid
                real_action = action

            obs, reward, terminated, truncated, info = env.step(real_action)
            done = terminated or truncated

            total_reward += reward
            episode_times.append(step)

            # PID needs torque; RL uses action index
            if controller_type == "pid":
                episode_torques.append(action)  # PID torque
            else:
                episode_torques.append(1 if action == 1 else -1)  # Assume discrete action maps to left(-1)/right(+1)

            episode_angles.append(obs[2])  # Assuming obs[2] is pole angle

            if done:
                if abs(obs[2]) > 0.8:  # Assume 0.8 rad (~45 deg) as falling condition
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
    parser.add_argument('--model_path', type=str, required=True, help="Path to model or Q-table")
    parser.add_argument('--controller_type', type=str, choices=['qlearning', 'dqn', 'pid'], required=True)
    parser.add_argument('--use_wandb', action='store_true', help="Enable wandb logging")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="assistive-walker-eval",
            config=vars(config),
            name=f"eval_{args.controller_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    # 🎮 Setup env
    env = gymnasium.make("CartPole-v1", render_mode="human")
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    np.random.seed(config.seed)

    # 📦 Load controller
    if args.controller_type == "qlearning":
        with open(args.model_path, "rb") as f:
            q_table = pickle.load(f)
        from train.utils.discretizer import create_bins
        obs_low, obs_high = env.observation_space.low, env.observation_space.high
        bins = create_bins(obs_low, obs_high, config.bins)
        controller = RLAgent(q_table, bins, obs_low, obs_high)

    elif args.controller_type == "dqn":
        model = torch.load(args.model_path)
        controller = DRLController(model)

    elif args.controller_type == "pid":
        controller = PIDController(config.Kp, config.Ki, config.Kd)

    else:
        raise ValueError("Unknown controller type")

    logger.info(f"🔍 Evaluating {args.controller_type} controller...")

    # 🧪 Evaluate
    rewards, metrics, overshoots, settling_times, energy, smoothness = evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
    mean_reward = np.mean(rewards)
    # Example: Calling the evaluate_controller function)

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
    env.close()
    if args.use_wandb:
        wandb.finish()
