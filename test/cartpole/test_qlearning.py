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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole import CartPoleDiscreteEnv
from controllers.qlearning_controller import QLearningController
from train.utils.logger import setup_logger

logger = setup_logger()

register(
    id='CustomCartPole',
    entry_point='environments.cartpole:CartPoleDiscreteEnv',
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

def get_convergence_time(rewards, threshold=0.95):
    for episode, reward in enumerate(rewards):
        if reward >= threshold:
            return episode
    return len(rewards)

def get_avg_reward(rewards):
    return float(np.mean(rewards))

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
    for t, a in zip(times, angles):
        if abs(a - target_angle) <= tolerance:
            return float(t)
    return float(max(times)) if times else 0.0

def get_fall_rate(falls, total):
    return float((falls / total) * 100)

def get_energy(forces):
    flat = [abs(x) for x in forces]
    return float(sum(flat))

def get_smoothness(values, steps):
    if len(steps) > 1:
        jerks = np.diff(values) / np.diff(steps)
        return float(np.mean(np.abs(jerks)))
    return 0.0

def get_robustness(rewards):
    return float(np.std(rewards))

def evaluate_q_agent(env, agent, num_episodes, max_steps):
    rewards, overshoots, settling_times = [], [], []
    energy_list, smoothness_list = [], []
    fall_count = 0

    # For logging all obs and actions
    all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions = [], [], [], [], []

    bins = [
        np.linspace(-2.4, 2.4, 10),  # Cart position
        np.linspace(-3.0, 3.0, 10),  # Cart velocity
        np.linspace(-0.2, 0.2, 10),  # Pole angle
        np.linspace(-2.0, 2.0, 10),  # Pole velocity
    ]

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        state = [np.digitize(obs[i], bins[i]) for i in range(len(obs))]
        q_table_shape = (len(bins), 2)
        next_state = state
        min_bound = np.zeros_like(next_state)
        max_bound = q_table_shape - 1
        state_idx = tuple(np.clip(next_state, min_bound, max_bound).astype(int))

        total_reward = 0
        actions, cart_pos, cart_vel, pole_angle, pole_vel, steps = [], [], [], [], [], []

        for step in range(max_steps):
            if len(state_idx) != len(agent.q_table.shape[:-1]):
                break
            action = int(np.argmax(agent.q_table[state_idx]))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Record all obs and action at this step
            cart_pos.append(next_obs[0])
            cart_vel.append(next_obs[1])
            pole_angle.append(next_obs[2])
            pole_vel.append(next_obs[3])
            actions.append(action)
            steps.append(step)

            next_state = [np.digitize(next_obs[i], bins[i]) for i in range(len(next_obs))]
            next_state = np.array(next_state)
            state_idx = tuple(np.clip(next_state, min_bound, max_bound).astype(int))

            total_reward += reward

            if done:
                if abs(next_obs[2]) > 0.8:
                    fall_count += 1
                break

        # Store per-episode time series for summary stats
        all_cart_pos.append(cart_pos)
        all_cart_vel.append(cart_vel)
        all_pole_angle.append(pole_angle)
        all_pole_vel.append(pole_vel)
        all_actions.append(actions)

        rewards.append(total_reward)
        overshoots.append(get_overshoot(pole_angle) if pole_angle else 0)
        settling_times.append(get_settling_time(steps, pole_angle) if pole_angle else max_steps)
        energy_list.append(get_energy(actions))
        smoothness_list.append(get_smoothness(pole_angle, steps))

    # Compute per-episode mean/max/min/std for all obs and actions
    def ep_stats(series):
        return {
            "mean": float(np.mean([np.mean(ep) if ep else 0 for ep in series])),
            "max": float(np.max([np.max(ep) if ep else 0 for ep in series])),
            "min": float(np.min([np.min(ep) if ep else 0 for ep in series])),
            "std": float(np.std([np.std(ep) if ep else 0 for ep in series]))
        }

    metrics = {
        "Convergence Time (ep)": get_convergence_time(rewards),
        "Avg Reward": get_avg_reward(rewards),
        "Overshoot (°)": float(np.mean(overshoots)),
        "Settling Time (steps)": float(np.mean(settling_times)),
        "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
        "Energy (∑|F|)": float(np.mean(energy_list)),
        "Smoothness (Jerk)": float(np.mean(smoothness_list)),
        "Robustness": get_robustness(rewards),
        # Observation and action stats
        "CartPos Mean": ep_stats(all_cart_pos)["mean"],
        "CartPos Max": ep_stats(all_cart_pos)["max"],
        "CartPos Min": ep_stats(all_cart_pos)["min"],
        "CartPos Std": ep_stats(all_cart_pos)["std"],
        "CartVel Mean": ep_stats(all_cart_vel)["mean"],
        "CartVel Max": ep_stats(all_cart_vel)["max"],
        "CartVel Min": ep_stats(all_cart_vel)["min"],
        "CartVel Std": ep_stats(all_cart_vel)["std"],
        "PoleAngle Mean": ep_stats(all_pole_angle)["mean"],
        "PoleAngle Max": ep_stats(all_pole_angle)["max"],
        "PoleAngle Min": ep_stats(all_pole_angle)["min"],
        "PoleAngle Std": ep_stats(all_pole_angle)["std"],
        "PoleVel Mean": ep_stats(all_pole_vel)["mean"],
        "PoleVel Max": ep_stats(all_pole_vel)["max"],
        "PoleVel Min": ep_stats(all_pole_vel)["min"],
        "PoleVel Std": ep_stats(all_pole_vel)["std"],
        "Action Mean": ep_stats(all_actions)["mean"],
        "Action Max": ep_stats(all_actions)["max"],
        "Action Min": ep_stats(all_actions)["min"],
        "Action Std": ep_stats(all_actions)["std"],
    }

    return rewards, metrics, overshoots, settling_times, energy_list, smoothness_list, all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions

# 🚀 Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    env = CartPoleDiscreteEnv(render_mode=True, test_mode=True)

    # Set seeds for reproducibility
    env.action_space.seed(getattr(config, 'seed', 42))
    env.observation_space.seed(getattr(config, 'seed', 42))
    np.random.seed(getattr(config, 'seed', 42))
    random.seed(getattr(config, 'seed', 42))

    # Load Q-table
    with open(args.model_path, "rb") as f:
        q_table = pickle.load(f)
    agent = QLearningController(q_table)

    if args.use_wandb:
        wandb.init(
            project="Q-Learning-CartPole-test",
            config=vars(config),
            name=f"eval_Qlearning_{datetime.now():%Y%m%d_%H%M%S}",
        )

    # Use config for episodes and max_steps if present, else defaults
    num_episodes = getattr(config, "eval_episodes", 100)
    max_steps = getattr(config, "max_steps", 200)

    # Evaluate the agent
    results = evaluate_q_agent(env, agent, num_episodes=num_episodes, max_steps=max_steps)
    rewards, metrics, overshoots, settling_times, energy_list, smoothness_list, all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions = results

    # Print summary stats for obs and forces
    print("==== Observation and Action Statistics ====")
    for k in ["CartPos", "CartVel", "PoleAngle", "PoleVel", "Action"]:
        print(f"{k} Mean: {metrics[k+' Mean']:.3f}, Max: {metrics[k+' Max']:.3f}, Min: {metrics[k+' Min']:.3f}, Std: {metrics[k+' Std']:.3f}")

    if args.use_wandb:
        data = []
        for ep in range(len(rewards)):
            data.append([
                ep,
                float(rewards[ep]),
                float(overshoots[ep]),
                float(settling_times[ep]),
                float(energy_list[ep]),
                float(smoothness_list[ep]),
                float(np.mean(all_cart_pos[ep]) if all_cart_pos[ep] else 0),
                float(np.mean(all_cart_vel[ep]) if all_cart_vel[ep] else 0),
                float(np.mean(all_pole_angle[ep]) if all_pole_angle[ep] else 0),
                float(np.mean(all_pole_vel[ep]) if all_pole_vel[ep] else 0),
                float(np.mean(all_actions[ep]) if all_actions[ep] else 0),
            ])
        columns = [
            "episode", "reward", "overshoot", "settling_time", "energy", "smoothness",
            "mean_cart_pos", "mean_cart_vel", "mean_pole_angle", "mean_pole_vel", "mean_action"
        ]
        table = wandb.Table(data=data, columns=columns)
        wandb.log({
            "mean_reward": float(np.mean(rewards)),
            **{k: float(v) for k, v in metrics.items()},
            "Reward Curve": wandb.plot.line(table, "episode", "reward", title="Reward per Episode"),
            "Cart Position Curve": wandb.plot.line(table, "episode", "mean_cart_pos", title="Cart Position per Episode"),
            "Cart Velocity Curve": wandb.plot.line(table, "episode", "mean_cart_vel", title="Cart Velocity per Episode"),
            "Pole Angle Curve": wandb.plot.line(table, "episode", "mean_pole_angle", title="Pole Angle per Episode"),
            "Pole Velocity Curve": wandb.plot.line(table, "episode", "mean_pole_vel", title="Pole Velocity per Episode"),
            "Action Curve": wandb.plot.line(table, "episode", "mean_action", title="Action per Episode"),
            "Overshoot Curve": wandb.plot.line(table, "episode", "overshoot", title="Overshoot per Episode"),
            "Settling Time Curve": wandb.plot.line(table, "episode", "settling_time", title="Settling Time per Episode"),
            "Energy Curve": wandb.plot.line(table, "episode", "energy", title="Energy per Episode"),
            "Smoothness Curve": wandb.plot.line(table, "episode", "smoothness", title="Smoothness per Episode"),
            "evaluation_table": table
        })
        wandb.finish()
