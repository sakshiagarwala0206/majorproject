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
import pybullet as p

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environments.walker import WalkerBalanceContinuousEnv
from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import PPO

logger = setup_logger()

# Register the custom environment
register(
    id="WalkerBalanceContinuousEnv-v0",
    entry_point="environments.walker:WalkerBalanceContinuousEnv",
    max_episode_steps=500,
)

# Load config from YAML
def load_config(path):
    with open(path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

# Metric functions
def get_avg_reward(rewards): return float(np.mean(rewards))
def get_robustness(rewards): return float(np.std(rewards))
def get_energy(torques): return float(sum(np.abs(np.concatenate(torques))))
def get_smoothness(angles, steps): return float(np.mean(np.abs(np.diff(angles) / np.diff(steps))))
def get_fall_rate(falls, total): return float((falls / total) * 100)

# Load robot URDF
def load_robot():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")

    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
    return robot_id

# Additional metrics
def get_overshoot(angles): 
    return max(abs(np.diff(angles)))  # simple overshoot metric

def get_settling_time(angles, steps): 
    settling_time = None
    for i in range(1, len(angles)):
        if abs(angles[i] - angles[0]) < 0.1:
            settling_time = steps[i]
            break
    return settling_time if settling_time else steps[-1]

# Evaluation function
def evaluate_controller(env, controller, episodes, max_steps):
    rewards, energy_list, smoothness_list, overshoots, settling_times = [], [], [], [], []
    fall_count = 0

    # Load the robot in the environment
    robot_id = load_robot()  # Now calling load_robot() here

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0
        torques, angles, steps = [], [], []

        for step in range(max_steps):
            action, _ = controller.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Log the position and orientation at each step for debugging
            position, orientation = p.getBasePositionAndOrientation(robot_id)
            print(f"Position: {position}, Orientation: {orientation}")
            
            # Log reaction forces at each step for debugging
            reaction_forces = p.getContactPoints(robot_id)
            print(f"Reaction Forces: {reaction_forces}")

            contact_points = p.getContactPoints(robot_id)
            if contact_points:
                for contact in contact_points:
                    print(f"Contact point: {contact}")
            
            total_reward += reward
            torques.append(action)
            angles.append(obs[2])
            steps.append(step)

            if done:
                if abs(obs[2]) > 0.8: fall_count += 1
                break

        rewards.append(total_reward)
        energy_list.append(get_energy(torques))
        smoothness_list.append(get_smoothness(angles, steps))
        overshoots.append(get_overshoot(angles))
        settling_times.append(get_settling_time(angles, steps))

    metrics = {
        "Avg Reward": get_avg_reward(rewards),
        "Fall Rate (%)": get_fall_rate(fall_count, episodes),
        "Energy (∑|τ|)": float(np.mean(energy_list)),
        "Smoothness (Jerk)": float(np.mean(smoothness_list)),
        "Robustness": get_robustness(rewards)
    }

    return rewards, overshoots, settling_times, energy_list, smoothness_list, metrics


def log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config):
    data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
    cols = ["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"]
    episode_table = wandb.Table(data=data, columns=cols)

    # Log the evaluation metrics
    wandb.log({"mean_reward": np.mean(rewards), **metrics})

    # Log plots
    plots = {
        "Reward Curve": ("reward", "Reward per Episode"),
        "Overshoot Curve": ("overshoot", "Overshoot per Episode"),
        "Settling Time Curve": ("settling_time", "Settling Time per Episode"),
        "Energy Curve": ("energy", "Energy per Episode"),
        "Smoothness Curve": ("smoothness", "Smoothness per Episode"),
    }

    for key, (y_col, title) in plots.items():
        wandb.log({
            key: wandb.plot.line(episode_table, "episode", y_col, title=title)
        })

    wandb.log({"Evaluation Data": episode_table})


# Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    # WandB setup
    if args.use_wandb:
        wandb.init(
            project="assistive-walker-eval-DRL",
            config=vars(config),
            name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
        )

    # Initialize environment and seeds
    env = gymnasium.make("WalkerBalanceContinuousEnv-v0", render_mode="True")
    env.action_space.seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # Load model and controller
    model = PPO.load(args.model_path)
    controller = DRLController(model, "PPO", env.action_space)

    # Evaluate
    logger.info("Evaluating WalkerBalanceContinuousEnv with PPO...")
    rewards, overshoots, settling_times, energy_list, smoothness_list, metrics = evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
    logger.info(f"Results: {metrics}")

    # Log to WandB after evaluation
    if args.use_wandb:
        log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config)
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
#     max_episode_steps=500,
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
# def load_robot(self):
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#         urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF file not found at {urdf_path}")

#         self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])
#         return self.robot_id

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

# # # Evaluation function
# # def evaluate_controller(env, controller, episodes, max_steps):
# #     rewards, energy_list, smoothness_list, overshoots, settling_times = [], [], [], [], []
# #     fall_count = 0

# #     for ep in range(episodes):
# #         obs, _ = env.reset(seed=ep)
# #         total_reward = 0
# #         torques, angles, steps = [], [], []

# #         for step in range(max_steps):
# #             action, _ = controller.model.predict(obs, deterministic=True)
# #             obs, reward, terminated, truncated, _ = env.step(action)
# #             done = terminated or truncated
# #             # Log the position and orientation at each step for debugging
# #             position, orientation = p.getBasePositionAndOrientation(robot_id)  # Replace with your robot_id
# #             print(f"Position: {position}, Orientation: {orientation}")
            
# #             # Log reaction forces at each step for debugging
# #             reaction_forces = p.getContactPoints(robot_id)  # Replace with your robot_id
# #             print(f"Reaction Forces: {reaction_forces}")

# #             total_reward += reward
# #             torques.append(action)
# #             angles.append(obs[2])
# #             steps.append(step)

# #             if done:
# #                 if abs(obs[2]) > 0.8: fall_count += 1
# #                 break

# #         rewards.append(total_reward)
# #         energy_list.append(get_energy(torques))
# #         smoothness_list.append(get_smoothness(angles, steps))
# #         overshoots.append(get_overshoot(angles))
# #         settling_times.append(get_settling_time(angles, steps))

# #     metrics = {
# #         "Avg Reward": get_avg_reward(rewards),
# #         "Fall Rate (%)": get_fall_rate(fall_count, episodes),
# #         "Energy (∑|τ|)": float(np.mean(energy_list)),
# #         "Smoothness (Jerk)": float(np.mean(smoothness_list)),
# #         "Robustness": get_robustness(rewards)
# #     }

# #     return rewards, overshoots, settling_times, energy_list, smoothness_list, metrics

# def evaluate_controller(env, controller, episodes, max_steps):
#     rewards, energy_list, smoothness_list, overshoots, settling_times = [], [], [], [], []
#     fall_count = 0

#     # Ensure PyBullet is connected and robot_id is defined
#     robot_id = env.robot_id  # Assuming the robot_id is stored in the environment

#     for ep in range(episodes):
#         obs, _ = env.reset(seed=ep)
#         total_reward = 0
#         torques, angles, steps = [], [], []

#         for step in range(max_steps):
#             action, _ = controller.model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             # Log the position and orientation at each step for debugging
#             position, orientation = p.getBasePositionAndOrientation(robot_id)  # Replace with your robot_id
#             print(f"Position: {position}, Orientation: {orientation}")
            
#             # Log reaction forces at each step for debugging
#             reaction_forces = p.getContactPoints(robot_id)  # Replace with your robot_id
#             print(f"Reaction Forces: {reaction_forces}")
            
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
#     env = gymnasium.make("WalkerBalanceContinuousEnv-v0", render_mode="human")
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
