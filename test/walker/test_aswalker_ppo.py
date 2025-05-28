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
import pybullet as p

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environments.walker import AssistiveWalkerContinuousEnv
from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import PPO

logger = setup_logger()

register(
    id="WalkerBalanceContinuousEnv-v0",
    entry_point="environments.walker:AssistiveWalkerContinuousEnv",
    max_episode_steps=1000,
)

def load_config(path):
    with open(path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

def load_robot():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
    return robot_id

# --- Metric Functions (same as CartPole) ---
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
        jerks = np.diff(angles) / (np.diff(steps) * dt)
        if len(jerks) > 0 and not np.any(np.isnan(jerks)):
            return float(np.mean(np.abs(jerks)))
    return 0.0

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(steps, angles, tolerance=0.1, disturbance_step=200):
#     settled = False
#     for i, (s, a) in enumerate(zip(steps, angles)):
#         if s >= disturbance_step and abs(a) <= tolerance:
#             # Optionally: check if it remains within tolerance for N steps
#             return float(s - disturbance_step)
#     return float(steps[-1] - disturbance_step) if steps else 0.0
def get_settling_time(steps, angles, tolerance=0.1):
    for i, a in enumerate(angles):
        if abs(a) <= tolerance:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0

def evaluate_controller(env, controller, episodes, max_steps):
    log_data = {
        "episode": [], "reward": [], "overshoot": [], "settling_time": [],
        "smoothness": [], "energy": [], "fall_rate": [], "episode_length": []
    }
    fall_count = 0
    robot_id = load_robot()

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        episode_forces = []
        episode_angles = []
        episode_steps = []
        done = False
        fall_flag = False

        for step in range(max_steps):
            # Disturbances
            if step == 100:
                base_mass = p.getDynamicsInfo(robot_id, -1)[0]
                p.changeDynamics(robot_id, -1, mass=base_mass + 20)
            if step == 200:
                p.applyExternalForce(
                    objectUniqueId=robot_id, linkIndex=-1,
                    forceObj=[4, 0, 0], posObj=[0, 0, 0], flags=p.WORLD_FRAME
                )

            action, _ = controller.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # For walker, use e.g. obs[2] as the main angle (adapt if needed)
            episode_angles.append(obs[2])
            episode_forces.append(action)
            episode_steps.append(step)
            total_reward += reward

            if done:
                if abs(obs[2]) > 0.8 or step < max_steps - 1:
                    fall_flag = True
                    fall_count += 1
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

        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_rate}")
    import pandas as pd
    # Aggregate metrics
    metrics = ["reward", "overshoot", "settling_time", "smoothness", "energy", "fall_rate", "episode_length"]
    df = pd.DataFrame(log_data)
    df.to_csv("walker_ppo_metrics.csv", index=False)
    print("Saved metrics to walker_ppo_metrics.csv")

    summary = {f"{m}_mean": float(df[m].mean()) for m in metrics}
    summary.update({f"{m}_std": float(df[m].std()) for m in metrics})
    summary.update({f"{m}_median": float(df[m].median()) for m in metrics})

    return df, summary

def log_to_wandb(df, summary):
    episode_table = wandb.Table(dataframe=df)
    wandb.log(summary)
    for metric in df.columns:
        if metric != "episode":
            wandb.log({
                f"{metric}_curve": wandb.plot.line(
                    episode_table, "episode", metric, title=f"{metric.replace('_', ' ').title()} Over Episodes"
                )
            })
    wandb.log({"Evaluation Data": episode_table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="assistive-walker-test",
            config=vars(config),
            name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
        )

    env = gym.make("WalkerBalanceContinuousEnv-v0", render_mode="human")
    env.action_space.seed(getattr(config, "seed", 0))
    np.random.seed(getattr(config, "seed", 0))
    random.seed(getattr(config, "seed", 0))

    model = PPO.load(args.model_path)
    controller = DRLController(model, "PPO", action_space=env.action_space)

    df, summary = evaluate_controller(
        env, controller, getattr(config, "eval_episodes", 100), getattr(config, "max_steps", 1000)
    )

    if args.use_wandb:
        log_to_wandb(df, summary)
        wandb.finish()

    env.close()













# import argparse
# import gymnasium
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
# import pybullet as p

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from environments.walker import WalkerBalanceContinuousEnv
# from controllers.drl_controller import DRLController
# from train.utils.logger import setup_logger
# from stable_baselines3 import PPO

# logger = setup_logger()

# # Register the custom environment
# register(
#     id="WalkerBalanceContinuousEnv-v0",
#     entry_point="environments.walker:WalkerBalanceContinuousEnv",
#     max_episode_steps=1000,
# )

# # Load config from YAML
# def load_config(path):
#     with open(path, 'r') as f:
#         return SimpleNamespace(**yaml.safe_load(f))

# # Metric functions
# def get_avg_reward(rewards): return float(np.mean(rewards))
# def get_robustness(rewards): return float(np.std(rewards))
# def get_energy(torques): return float(sum(np.abs(np.concatenate(torques))))
# def get_smoothness(angles, steps): return float(np.mean(np.abs(np.diff(angles) / np.diff(steps))))
# def get_fall_rate(falls, total): return float((falls / total) * 100)

# # Load robot URDF
# def load_robot():
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#     urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#     if not os.path.exists(urdf_path):
#         raise FileNotFoundError(f"URDF file not found at {urdf_path}")

#     robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
#     return robot_id

# # Additional metrics
# def get_overshoot(angles): 
#     return max(abs(np.diff(angles)))  # simple overshoot metric

# def get_settling_time(angles, steps): 
#     settling_time = None
#     for i in range(1, len(angles)):
#         if abs(angles[i] - angles[0]) < 0.1:
#             settling_time = steps[i]
#             break
#     return settling_time if settling_time else steps[-1]

# # Evaluation function
# def evaluate_controller(env, controller, episodes, max_steps):
#     rewards, energy_list, smoothness_list, overshoots, settling_times = [], [], [], [], []
#     fall_count = 0

#     # Load the robot in the environment
#     robot_id = load_robot()  # Now calling load_robot() here

#     for ep in range(episodes):
#         obs, _ = env.reset(seed=ep)
#         total_reward = 0
#         torques, angles, steps = [], [], []

#         for step in range(max_steps):

#                     # Sudden payload increase
#             if step == 100:
#                 base_mass = p.getDynamicsInfo(robot_id, -1)[0]
#                 p.changeDynamics(robot_id, -1, mass=base_mass + 20)

#             if step == 200:
#                 # Apply a force in the x-direction (lateral to the walker base)
#                 force_vector = [4, 0, 0]  # 4 N along x-axis
#                 position = [0, 0, 0]      # Apply at the base of the walker
#                 p.applyExternalForce(
#                     objectUniqueId=robot_id,  # walker base unique ID
#                     linkIndex=-1,             # -1 for base link
#                     forceObj=force_vector,
#                     posObj=position,
#                     flags=p.WORLD_FRAME
#                 )
                
#             action, _ = controller.model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             # Log the position and orientation at each step for debugging
#             position, orientation = p.getBasePositionAndOrientation(robot_id)
#             print(f"Position: {position}, Orientation: {orientation}")
            
#             # Log reaction forces at each step for debugging
#             reaction_forces = p.getContactPoints(robot_id)
#             print(f"Reaction Forces: {reaction_forces}")

#             contact_points = p.getContactPoints(robot_id)
#             if contact_points:
#                 for contact in contact_points:
#                     print(f"Contact point: {contact}")
            
#             total_reward += reward
#             torques.append(action)
#             angles.append(obs[2])
#             steps.append(step)

#             if done:
#                 if abs(obs[2]) > 0.8: fall_count += 1
#                 break

#         rewards.append(total_reward)
#         energy_list.append(get_energy(torques))
#         smoothness_list.append(get_smoothness(angles, steps))
#         overshoots.append(get_overshoot(angles))
#         settling_times.append(get_settling_time(angles, steps))

#     metrics = {
#         "Avg Reward": get_avg_reward(rewards),
#         "Fall Rate (%)": get_fall_rate(fall_count, episodes),
#         "Energy (∑|τ|)": float(np.mean(energy_list)),
#         "Smoothness (Jerk)": float(np.mean(smoothness_list)),
#         "Robustness": get_robustness(rewards)
#     }

#     return rewards, overshoots, settling_times, energy_list, smoothness_list, metrics


# def log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config):
#     data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
#     cols = ["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"]
#     episode_table = wandb.Table(data=data, columns=cols)

#     # Log the evaluation metrics
#     wandb.log({"mean_reward": np.mean(rewards), **metrics})

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


# # Main
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)

#     # WandB setup
#     if args.use_wandb:
#         wandb.init(
#             project="assistive-walker-eval-DRL",
#             config=vars(config),
#             name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
#         )

#     # Initialize environment and seeds
#     env = gymnasium.make("WalkerBalanceContinuousEnv-v0", render_mode="True")
#     env.action_space.seed(config.seed)
#     np.random.seed(config.seed)
#     random.seed(config.seed)

#     # Load model and controller
#     model = PPO.load(args.model_path)
#     controller = DRLController(model, "PPO", env.action_space)

#     # Evaluate
#     logger.info("Evaluating WalkerBalanceContinuousEnv with PPO...")
#     rewards, overshoots, settling_times, energy_list, smoothness_list, metrics = evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
#     logger.info(f"Results: {metrics}")

#     # Log to WandB after evaluation
#     if args.use_wandb:
#         log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config)
#         wandb.finish()

#     env.close()



