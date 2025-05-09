import argparse
import gymnasium as gym
import numpy as np
import os
import pickle
import random
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb
from gymnasium.envs.registration import register
import sys

# 📁 Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole import CartPoleDiscreteEnv
from controllers.qlearning_controller import QLearningController
from train.utils.logger import setup_logger

logger = setup_logger()

# ✅ Register custom environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole:CartPoleDiscreteEnv',
)

# 📄 Load YAML config
def load_config(config_path):
    with open(config_path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

# 📊 Evaluation metrics
def get_convergence_time(rewards, threshold=0.95):
    for episode, reward in enumerate(rewards):
        if reward >= threshold:
            return episode
    return len(rewards)

def get_avg_reward(rewards):
    return float(np.mean(rewards))

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles))

def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
    for t, a in zip(times, angles):
        if abs(a - target_angle) <= tolerance:
            return float(t)
    return float(max(times))

def get_fall_rate(falls, total):
    return float((falls / total) * 100)

def get_energy(torques):
    flat = [abs(x) for vec in torques for x in vec]
    return float(sum(flat))

def get_smoothness(angles, steps):
    if len(steps) > 1:
        jerks = np.diff(angles) / np.diff(steps)
        return float(np.mean(np.abs(jerks)))
    return 0.0

def get_robustness(rewards):
    return float(np.std(rewards))


# 🎯 Main evaluation loop with random disturbances
def evaluate_q_agent(env, agent, num_episodes, max_steps, disturbance_factor=0.1):
    rewards, overshoots, settling_times = [], [], []
    energy_list, smoothness_list = [], []
    fall_count = 0

    # Discretization bins
    bins = [
        np.linspace(-2.4, 2.4, 10),  # Cart position
        np.linspace(-3.0, 3.0, 10),  # Cart velocity
        np.linspace(-0.2, 0.2, 10),  # Pole angle
        np.linspace(-2.0, 2.0, 10),  # Pole velocity
    ]
    q_table_shape = tuple(len(b) + 1 for b in bins)
    min_bound = np.zeros(len(q_table_shape), dtype=int)
    max_bound = np.array(q_table_shape) - 1

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)

        # Apply random disturbance to initial observation
        obs = np.array(obs) + np.random.uniform(-disturbance_factor, disturbance_factor, size=obs.shape)

        # Discretize initial state
        state = tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))
        state_idx = tuple(np.clip(state, min_bound, max_bound))

        total_reward = 0
        torques, angles, steps = [], [], []

        for step in range(max_steps):
            action = agent.act(state_idx)
            epsilon = 0.7  # Exploration factor

            if np.random.rand() < epsilon:
                action = np.random.choice([0, 1])  # Random action
            else:
                action = np.argmax(q_table[state_idx])  # Greedy action
                        # Apply random disturbance to the action
            # action += np.random.choice([-1, 1]) * np.random.uniform(0, disturbance_factor)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Apply random disturbance to the next observation
            next_obs = np.array(next_obs) + np.random.uniform(-disturbance_factor, disturbance_factor, size=next_obs.shape)

            next_state = tuple(np.digitize(next_obs[i], bins[i]) for i in range(len(next_obs)))
            state_idx = tuple(np.clip(next_state, min_bound, max_bound))

            torques.append([action])
            angles.append(next_obs[2])
            steps.append(step)
            total_reward += reward

            if done:
                if abs(next_obs[2]) > 0.8:
                    fall_count += 1
                break

        rewards.append(total_reward)
        if angles:
            overshoots.append(get_overshoot(angles))
            settling_times.append(get_settling_time(steps, angles))
            energy_list.append(get_energy(torques))
            smoothness_list.append(get_smoothness(angles, steps))
        else:
            overshoots.append(0)
            settling_times.append(max_steps)
            energy_list.append(0)
            smoothness_list.append(0)

    metrics = {
        "Convergence Time (ep)": get_convergence_time(rewards),
        "Avg Reward": get_avg_reward(rewards),
        "Overshoot (°)": float(np.mean(overshoots)),
        "Settling Time (steps)": float(np.mean(settling_times)),
        "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
        "Energy (∑|τ|)": float(np.mean(energy_list)),
        "Smoothness (Jerk)": float(np.mean(smoothness_list)),
        "Robustness": get_robustness(rewards)
    }

    return rewards, metrics, overshoots, settling_times, energy_list, smoothness_list



# 🚀 Entry point
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    if args.use_wandb:
        wandb.init(project="qlearning-evaluation", name="QAgent-Eval")

    config = load_config(args.config)
    env = CartPoleDiscreteEnv(render_mode=True)

    # Load or initialize q_table
    if os.path.exists(args.model_path):
        with open(args.model_path, 'rb') as f:
            q_table = pickle.load(f)
            print("Q-table type:", type(q_table))

    else:
        q_table = {}
        for cart_pos in range(11):
            for cart_vel in range(11):
                for pole_angle in range(11):
                    for pole_vel in range(11):
                        state = (cart_pos, cart_vel, pole_angle, pole_vel)
                        q_table[state] = [0] * 2

    agent = QLearningController(q_table)

    rewards, metrics, overshoots, settling_times, energy_list, smoothness_list = evaluate_q_agent(
        env, agent, num_episodes=100, max_steps=200
    )

    if args.use_wandb:
        data = [
            [ep, float(rewards[ep]), float(overshoots[ep]), float(settling_times[ep]),
             float(energy_list[ep]), float(smoothness_list[ep])]
            for ep in range(len(rewards))
        ]

        table = wandb.Table(data=data, columns=[
            "episode", "reward", "overshoot", "settling_time", "energy", "smoothness"
        ])

        wandb.log({
            "mean_reward": float(np.mean(rewards)),
            **{k: float(v) for k, v in metrics.items()},
            "Reward Curve": wandb.plot.line(table, "episode", "reward", title="Reward per Episode"),
            "Overshoot Curve": wandb.plot.line(table, "episode", "overshoot", title="Overshoot per Episode"),
            "Settling Time Curve": wandb.plot.line(table, "episode", "settling_time", title="Settling Time per Episode"),
            "Energy Curve": wandb.plot.line(table, "episode", "energy", title="Energy per Episode"),
            "Smoothness Curve": wandb.plot.line(table, "episode", "smoothness", title="Smoothness per Episode"),
            "evaluation_table": table
        })
        wandb.finish()

    env.close()
