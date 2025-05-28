# import argparse
# import gymnasium as gym
# import numpy as np
# import torch
# import os
# import sys
# import yaml
# from types import SimpleNamespace
# from datetime import datetime
# import wandb
# from gymnasium.envs.registration import register
# import random

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from controllers.drl_controller import DRLController
# from train.utils.logger import setup_logger
# from stable_baselines3 import PPO

# logger = setup_logger()

# # Register custom CartPole environment
# register(
#     id='CustomCartPole-v0',
#     entry_point='environments.cartpole:CartPoleContinuousEnv',
#     max_episode_steps=500,
# )

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config_dict = yaml.safe_load(f)
#     return SimpleNamespace(**config_dict)

# def get_statistics(data):
#     return {
#         "mean": float(np.mean(data)),
#         "std": float(np.std(data)),
#         "median": float(np.median(data))
#     }

# def get_fall_rate(falls, total):
#     return float((falls / total) * 100)

# def get_energy(forces):
#     flat = []
#     for v in forces:
#         try:
#             vals = list(v)
#         except TypeError:
#             vals = [v]
#         flat.extend([abs(x) for x in vals])
#     return float(sum(flat))

# def get_smoothness(angles, steps):
#     jerks = np.diff(angles) / np.diff(steps)
#     return float(np.mean(np.abs(jerks))) if len(jerks) > 0 else 0.0

# def get_overshoot(angles, target_angle=0.0):
#     return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
#     for t, a in zip(times, angles):
#         if abs(a - target_angle) <= tolerance:
#             return float(t)
#     return float(max(times)) if times else 0.0

# def evaluate_controller(env, controller, num_episodes, max_steps):
#     rewards, overshoots, settling_times, energy_list, smoothness_list = [], [], [], [], []
#     fall_count = 0

#     for ep in range(num_episodes):
#         obs, _ = env.reset(seed=ep)
#         total_reward = 0.0
#         episode_forces = []
#         episode_angles = []
#         episode_steps = []
#         done = False

#         for step in range(max_steps):
#             # Disturbance: Apply lateral impulse at step 200
#             if step == 200:
#                 if hasattr(env, "cartpole_id"):
#                     # Apply 4N force to cart base (PyBullet)
#                     import pybullet as p
#                     p.applyExternalForce(
#                         objectUniqueId=env.cartpole_id,
#                         linkIndex=-1,
#                         forceObj=[4, 0, 0],
#                         posObj=[0, 0, 0],
#                         flags=p.WORLD_FRAME
#                     )

#             action, _ = controller.model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             # obs: [pole_angle, pole_vel, cart_pos, cart_vel]
#             episode_angles.append(obs[0])
#             episode_forces.append(action)
#             episode_steps.append(step)
#             total_reward += reward

#             if done:
#                 if abs(obs[0]) > 0.8:  # Pole fell
#                     fall_count += 1
#                 break

#         rewards.append(total_reward)
#         energy_list.append(get_energy(episode_forces))
#         smoothness_list.append(get_smoothness(episode_angles, episode_steps))
#         overshoots.append(get_overshoot(episode_angles))
#         settling_times.append(get_settling_time(episode_steps, episode_angles))

#     metrics = {
#         "Avg Reward": get_statistics(rewards),
#         "Fall Rate (%)": {
#             "mean": get_fall_rate(fall_count, num_episodes),
#             "std": 0.0,
#             "median": get_fall_rate(fall_count, num_episodes)
#         },
#         "Energy (∑|τ|)": get_statistics(energy_list),
#         "Smoothness (Jerk)": get_statistics(smoothness_list),
#         "Overshoot": get_statistics(overshoots),
#         "Settling Time": get_statistics(settling_times)
#     }

#     print("\n==== CartPole Evaluation Summary ====")
#     for metric, stats in metrics.items():
#         print(f"{metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, median={stats['median']:.2f}")
#     print("=====================================\n")

#     return rewards, overshoots, settling_times, energy_list, smoothness_list, metrics

# def log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config):
#     data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
#     cols = ["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"]
#     episode_table = wandb.Table(data=data, columns=cols)

#     # Log the evaluation metrics
#     wandb.log({
#         "mean_reward": metrics["Avg Reward"]["mean"],
#         "std_reward": metrics["Avg Reward"]["std"],
#         "median_reward": metrics["Avg Reward"]["median"],
#         "fall_rate_mean": metrics["Fall Rate (%)"]["mean"],
#         "energy_mean": metrics["Energy (∑|τ|)"]["mean"],
#         "smoothness_mean": metrics["Smoothness (Jerk)"]["mean"],
#         "overshoot_mean": metrics["Overshoot"]["mean"],
#         "settling_time_mean": metrics["Settling Time"]["mean"],
#     })

#     # Log plots
#     plots = {
#         "Reward Curve": ("reward", "Reward per Episode"),
#         "Overshoot Curve": ("overshoot", "Overshoot per Episode"),
#         "Settling Time Curve": ("settling_time", "Settling Time per Episode"),
#         "Energy Curve": ("energy", "Energy per Episode"),
#         "Smoothness Curve": ("smoothness", "Smoothness per Episode"),
#     }

#     for key, (y_col, title) in plots.items():
#         wandb.log({
#             key: wandb.plot.line(episode_table, "episode", y_col, title=title)
#         })

#     wandb.log({"Evaluation Data": episode_table})

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)

#     if args.use_wandb:
#         wandb.init(
#             project="Cartpole_test_final",
#             config=vars(config),
#             name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
#         )

#     env = gym.make("CustomCartPole-v0", render_mode="human", test_mode=True)
#     env.action_space.seed(getattr(config, "seed", 0))
#     np.random.seed(getattr(config, "seed", 0))
#     random.seed(getattr(config, "seed", 0))

#     model = PPO.load(args.model_path)
#     controller = DRLController(model, "PPO", action_space=env.action_space)

#     logger.info(f"Evaluating PPO controller on CartPole...")

#     rewards, overshoots, settling_times, energy_list, smoothness_list, metrics = evaluate_controller(
#         env, controller, getattr(config, "eval_episodes", 30), getattr(config, "max_steps", 1000)
#     )

#     logger.info(f"Results: {metrics}")

#     if args.use_wandb:
#         log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config)
#         wandb.finish()

#     env.close()

import argparse
import gymnasium as gym
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
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import PPO

logger = setup_logger()

register(
    id='CustomCartPole-v0',
    entry_point='environments.cartpole:CartPoleContinuousEnv',
    max_episode_steps=500,
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def get_energy(forces):
    flat = []
    for v in forces:
        try:
            vals = list(v)
        except TypeError:
            vals = [v]
        flat.extend([abs(x) for x in vals])
    return float(sum(flat))

def get_smoothness(angles, steps, dt=1.0):
    if len(angles) > 1 and len(steps) > 1:
        jerks = np.diff(angles) / np.diff(steps)
        if len(jerks) > 0 and not np.any(np.isnan(jerks)):
            return float(np.mean(np.abs(jerks)))
    return 0.0

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

def get_settling_time(steps, angles, tolerance=0.1):
    """
    Returns the first time (step index) at which the angle enters and remains within the tolerance band.
    If it never settles, returns the episode length.
    """
    for i, a in enumerate(angles):
        if abs(a) <= tolerance:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="Cartpole_test_final",
            config=vars(config),
            name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
        )

    env = gym.make("CustomCartPole-v0", render_mode="human", test_mode=True)
    env.action_space.seed(getattr(config, "seed", 0))
    np.random.seed(getattr(config, "seed", 0))
    random.seed(getattr(config, "seed", 0))

    model = PPO.load(args.model_path)
    controller = DRLController(model, "PPO", action_space=env.action_space)

    max_episodes = getattr(config, "eval_episodes", 100)
    max_steps = getattr(config, "max_steps", 500)

    log_data = {
        "episode": [], "reward": [], "overshoot": [], "settling_time": [],
        "smoothness": [], "energy": [], "fall_rate": [], "episode_length": []
    }

    for ep in range(max_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        episode_forces = []
        episode_angles = []
        episode_steps = []
        done = False
        fall_flag = False

        for step in range(max_steps):
            if step == 200:
                if hasattr(env, "cartpole_id"):
                    import pybullet as p
                    p.applyExternalForce(
                        objectUniqueId=env.cartpole_id,
                        linkIndex=-1,
                        forceObj=[4, 0, 0],
                        posObj=[0, 0, 0],
                        flags=p.WORLD_FRAME
                    )

            action, _ = controller.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            episode_angles.append(obs[0])
            episode_forces.append(action)
            episode_steps.append(step)
            total_reward += reward

            if done:
                if abs(obs[0]) > 0.8 or step < max_steps - 1:
                    fall_flag = True
                break

        overshoot = get_overshoot(episode_angles)
        settling_time = get_settling_time(episode_steps, episode_angles)
        smoothness = get_smoothness(episode_angles, episode_steps, dt=1.0)
        energy = get_energy(episode_forces)
        episode_length = step + 1
        fall_rate = 1 if fall_flag else 0

        log_data["episode"].append(ep)
        log_data["reward"].append(total_reward)
        log_data["overshoot"].append(overshoot)
        log_data["settling_time"].append(settling_time)
        log_data["smoothness"].append(smoothness)
        log_data["energy"].append(energy)
        log_data["fall_rate"].append(fall_rate)
        log_data["episode_length"].append(episode_length)

        if args.use_wandb:
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "overshoot": overshoot,
                "settling_time": settling_time,
                "smoothness": smoothness,
                "energy": energy,
                "fall_rate": fall_rate,
                "episode_length": episode_length
            })

        print(f"Episode {ep+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_rate}")

    env.close()

    df = pd.DataFrame(log_data)
    df.to_csv("ppo_metrics.csv", index=False)
    print("Saved metrics to ppo_metrics.csv")

    metrics = ["reward", "overshoot", "settling_time", "smoothness", "energy", "fall_rate", "episode_length"]
    episode_table = wandb.Table(dataframe=df) if args.use_wandb else None

    if args.use_wandb:
        # Log summary statistics for each metric
        for metric in metrics:
            wandb.log({
                f"{metric}_mean": float(df[metric].mean()),
                f"{metric}_std": float(df[metric].std()),
                f"{metric}_median": float(df[metric].median())
            })
        # Unified line plots for each metric
        for metric in metrics:
            wandb.log({
                f"{metric}_curve": wandb.plot.line(
                    episode_table, "episode", metric, title=f"{metric.replace('_', ' ').title()} Over Episodes"
                )
            })
        wandb.log({"Evaluation Data": episode_table})
        wandb.finish()


