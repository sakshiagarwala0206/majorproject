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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole import CartPoleContinuousEnv
from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import SAC

logger = setup_logger()

register(
    id='CustomCartPole',
    entry_point='environments.cartpole:CartPoleContinuousEnv',
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def get_convergence_time(rewards, threshold=0.95):
    for episode, reward in enumerate(rewards):
        if reward >= threshold:
            return episode
    return len(rewards)

def get_avg_reward(rewards):
    return np.mean(rewards)

def get_overshoot(angles, target_angle=0):
    return max(abs(angle - target_angle) for angle in angles) if angles else 0.0

def get_settling_time(times, angles, target_angle=0, tolerance=0.05):
    for t, angle in zip(times, angles):
        if abs(angle - target_angle) <= tolerance:
            return t
    return max(times) if times else 0

def get_fall_rate(falls, total_episodes):
    return (falls / total_episodes) * 100

def get_energy(torques):
    flat = []
    for v in torques:
        try:
            vals = list(v)
        except TypeError:
            vals = [v]
        flat.extend([abs(x) for x in vals])
    return np.sum(flat)

def get_smoothness(angles, time_steps):
    jerks = np.diff(angles) / np.diff(time_steps) if len(angles) > 1 else [0.0]
    return np.mean(np.abs(jerks)) if len(jerks) > 0 else 0.0

def get_robustness(performances):
    return np.std(performances)

def evaluate_controller(
    env, controller, num_episodes, max_steps,
    epsilon_start=1.0, epsilon_min=0.7, epsilon_decay=0.999
):
    rewards, overshoots, settling_times, energy_list, smoothness_list = [], [], [], [], []
    fall_count = 0
    epsilon = epsilon_start

    # For logging all obs and actions
    all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions = [], [], [], [], []

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        episode_forces = []
        episode_cart_pos = []
        episode_cart_vel = []
        episode_pole_angle = []
        episode_pole_vel = []
        episode_times = []

        for step in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action, _ = controller.model.predict(obs, deterministic=True)
            action_arr = np.array(action, dtype=float).flatten()

            obs, reward, terminated, truncated, info = env.step(action_arr)
            done = terminated or truncated

            # Record all obs and force at this step
            episode_cart_pos.append(obs[0])
            episode_cart_vel.append(obs[1])
            episode_pole_angle.append(obs[2])
            episode_pole_vel.append(obs[3])
            episode_forces.append(action_arr.tolist())
            episode_times.append(step)

            total_reward += reward

            if done:
                if abs(obs[2]) > 0.8:
                    fall_count += 1
                break

        # Log scalars for WandB default charts (once per episode)
        if wandb.run is not None:
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "overshoot": get_overshoot(episode_pole_angle),
                "settling_time": get_settling_time(episode_times, episode_pole_angle),
                "energy": get_energy(episode_forces),
                "smoothness": get_smoothness(episode_pole_angle, episode_times),
                "mean_cart_pos": float(np.mean(episode_cart_pos)) if episode_cart_pos else 0,
                "mean_cart_vel": float(np.mean(episode_cart_vel)) if episode_cart_vel else 0,
                "mean_pole_angle": float(np.mean(episode_pole_angle)) if episode_pole_angle else 0,
                "mean_pole_vel": float(np.mean(episode_pole_vel)) if episode_pole_vel else 0,
                "mean_action": float(np.mean([a[0] if isinstance(a, (list, np.ndarray)) else a for a in episode_forces])) if episode_forces else 0,
            })

        all_cart_pos.append(episode_cart_pos)
        all_cart_vel.append(episode_cart_vel)
        all_pole_angle.append(episode_pole_angle)
        all_pole_vel.append(episode_pole_vel)
        all_actions.append([a[0] if isinstance(a, (list, np.ndarray)) else a for a in episode_forces])

        rewards.append(total_reward)
        overshoots.append(get_overshoot(episode_pole_angle))
        settling_times.append(get_settling_time(episode_times, episode_pole_angle))
        energy_list.append(get_energy(episode_forces))
        smoothness_list.append(get_smoothness(episode_pole_angle, episode_times))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    # Compute per-episode mean for all obs and actions
    def ep_means(series):
        return [float(np.mean(ep)) if ep else 0.0 for ep in series]

    metrics = {
        "Convergence Time (ep)": get_convergence_time(rewards),
        "Avg Reward": get_avg_reward(rewards),
        "Overshoot (°)": float(np.mean(overshoots)),
        "Settling Time (steps)": float(np.mean(settling_times)),
        "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
        "Energy (∑|τ|)": float(np.mean(energy_list)),
        "Smoothness (Jerk)": float(np.mean(smoothness_list)),
        "Robustness": get_robustness(rewards),
        # Observation means
        "CartPos Mean": float(np.mean(ep_means(all_cart_pos))),
        "CartVel Mean": float(np.mean(ep_means(all_cart_vel))),
        "PoleAngle Mean": float(np.mean(ep_means(all_pole_angle))),
        "PoleVel Mean": float(np.mean(ep_means(all_pole_vel))),
        "Action Mean": float(np.mean(ep_means(all_actions))),
    }

    return (
        rewards, metrics, overshoots, settling_times, energy_list, smoothness_list,
        all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions
    )


# 🚀 Main testing entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to controller config file")
    parser.add_argument('--model_path', type=str, required=True, help="Path to SAC model (.zip)")
    parser.add_argument('--model_type', type=str, required=True, choices=['DQN', 'DDPG', 'SAC', 'PPO'], help="Model type (DQN, DDPG, SAC, PPO)")
    parser.add_argument('--use_wandb', action='store_true', help="Enable wandb logging")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="cartpole-eval-DRL",
            config=vars(config),
            name=f"eval_SAC_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        )

    env = gymnasium.make("CustomCartPole", render_mode="human", test_mode=True)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    model = SAC.load(args.model_path)
    model_type = args.model_type
    controller = DRLController(model, model_type, action_space=env.action_space)

    logger.info(f"🔍 Evaluating SAC controller...")

    (rewards, metrics, overshoots, settling_times, energy, smoothness,
     all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions) = \
        evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
    mean_reward = np.mean(rewards)

    logger.info(f"📊 Mean reward over {config.eval_episodes} episodes: {mean_reward}")

    if args.use_wandb:
        data = []
        for ep in range(len(rewards)):
            data.append([
                ep,
                float(rewards[ep]),
                float(overshoots[ep]),
                float(settling_times[ep]),
                float(energy[ep]),
                float(smoothness[ep]),
                float(np.mean(all_cart_pos[ep]) if all_cart_pos[ep] else 0),
                float(np.mean(all_cart_vel[ep]) if all_cart_vel[ep] else 0),
                float(np.mean(all_pole_angle[ep]) if all_pole_angle[ep] else 0),
                float(np.mean(all_pole_vel[ep]) if all_pole_vel[ep] else 0),
                float(np.mean(all_actions[ep]) if all_actions[ep] else 0),
            ])
        cols = [
            "episode", "reward", "overshoot", "settling_time", "energy", "smoothness",
            "mean_cart_pos", "mean_cart_vel", "mean_pole_angle", "mean_pole_vel", "mean_action"
        ]
        episode_table = wandb.Table(data=data, columns=cols)

        wandb.log({
            "mean_reward": mean_reward,
            **metrics,
            "Reward Curve": wandb.plot.line(episode_table, "episode", "reward", title="Reward per Episode"),
            "Overshoot Curve": wandb.plot.line(episode_table, "episode", "overshoot", title="Overshoot per Episode"),
            "Settling Time Curve": wandb.plot.line(episode_table, "episode", "settling_time", title="Settling Time per Episode"),
            "Energy Curve": wandb.plot.line(episode_table, "episode", "energy", title="Energy per Episode"),
            "Smoothness Curve": wandb.plot.line(episode_table, "episode", "smoothness", title="Smoothness per Episode"),
            "Cart Position Curve": wandb.plot.line(episode_table, "episode", "mean_cart_pos", title="Cart Position per Episode"),
            "Cart Velocity Curve": wandb.plot.line(episode_table, "episode", "mean_cart_vel", title="Cart Velocity per Episode"),
            "Pole Angle Curve": wandb.plot.line(episode_table, "episode", "mean_pole_angle", title="Pole Angle per Episode"),
            "Pole Velocity Curve": wandb.plot.line(episode_table, "episode", "mean_pole_vel", title="Pole Velocity per Episode"),
            "Action Curve": wandb.plot.line(episode_table, "episode", "mean_action", title="Action per Episode"),
            "evaluation_table": episode_table
        })
        wandb.finish()

    env.close()
