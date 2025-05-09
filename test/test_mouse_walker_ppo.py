import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium.envs.registration import register
from controllers.drl_controller import DRLController
from stable_baselines3 import PPO

# Register custom env
register(
    id="AssistiveWalkerContinuousEnv-v0",
    entry_point="environments.walker:AssistiveWalkerContinuousEnv",
    max_episode_steps=500,
)

def compute_metrics(angles, torques, timestamps):
    dt = np.diff(timestamps)
    dtheta = np.diff(angles)
    smoothness = float(np.mean(np.abs(dtheta / dt))) if len(dt) > 0 else 0.0
    overshoot = float(np.max(np.abs(angles)))
    tol = 0.05
    settled = np.where(np.abs(angles) < tol)[0]
    settling_time = float(timestamps[settled[0]]) if settled.size > 0 else float(timestamps[-1])
    energy = float(np.sum(np.abs(torques)))
    return overshoot, settling_time, smoothness, energy

def main(model_path):
    print("\n=== Starting mouse-driven test for Assistive Walker ===\n")
    # Load model and env
    model = PPO.load(model_path)
    env = gym.make("AssistiveWalkerContinuousEnv-v0", render_mode="human")
    base_env = env.unwrapped
    dt = getattr(base_env, "time_step", 1.0/240.0)
    controller = DRLController(model, "PPO", env.action_space)

    # You may need to adjust these indices for your URDF/joint setup!
    pole_joint_index = getattr(base_env, "pole_joint_index", 2)  # Example: 2nd/3rd joint is the pole

    try:
        while True:
            obs, _ = env.reset()
            print("Environment reset. Use the mouse in the PyBullet GUI to disturb the robot (e.g., click and drag the pole or base).")
            print("Waiting for disturbance...")

            # Wait for disturbance: detect large change in pole angle or joint torque
            prev_angle = obs[0]
            while True:
                # env.render()  # REMOVED: PyBullet GUI renders automatically
                time.sleep(0.001)
                # Option 1: Detect change in pole angle
                curr_angle = obs[0]
                if abs(curr_angle - prev_angle) > 0.1:  # threshold for disturbance
                    print("Disturbance detected (angle change)!")
                    break
                # Option 2: Detect external torque (if available)
                js = p.getJointState(base_env.robot_id, pole_joint_index)
                if abs(js[3]) > 1e-3:
                    print("Disturbance detected (external torque)!")
                    break
                prev_angle = curr_angle
                # Step with zero action to keep sim alive
                env.step(np.zeros(env.action_space.shape))

            # Record recovery
            angles, torques, times = [], [], []
            t0 = time.time()
            consec = 0
            while True:
                action, _ = controller.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                p.stepSimulation()
                now = time.time() - t0
                angles.append(obs[0])
                torques.append(np.linalg.norm(action))
                times.append(now)

                if abs(obs[0]) < 0.05:
                    consec += 1
                else:
                    consec = 0
                if consec >= 30 or now > 5.0 or done:
                    break
                time.sleep(dt)

            over, sett, smooth, energy = compute_metrics(angles, torques, times)
            print(f"Recovery metrics: Overshoot={over:.3f}, Settling={sett:.3f}s, Smoothness={smooth:.3f}, Energy={energy:.3f}\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
    finally:
        env.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args.model_path)



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
# import time

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from environments.walker import AssistiveWalkerContinuousEnv
# from controllers.drl_controller import DRLController
# from train.utils.logger import setup_logger
# from stable_baselines3 import PPO

# logger = setup_logger()

# # Register the custom environment
# register(
#     id="AssistiveWalkerContinuousEnv-v0",
#     entry_point="environments.walker:AssistiveWalkerContinuousEnv",
#     max_episode_steps=500,
# )

# def load_config(path):
#     with open(path, 'r') as f:
#         return SimpleNamespace(**yaml.safe_load(f))

# # Metric functions
# def get_avg_reward(rewards): return float(np.mean(rewards))
# def get_robustness(rewards): return float(np.std(rewards))
# def get_energy(torques): return float(sum(np.abs(np.concatenate(torques))))
# def get_smoothness(angles, steps): return float(np.mean(np.abs(np.diff(angles) / np.diff(steps))))
# def get_fall_rate(falls, total): return float((falls / total) * 100)

# def load_robot():
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#     urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#     if not os.path.exists(urdf_path):
#         raise FileNotFoundError(f"URDF file not found at {urdf_path}")

#     robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
#     return robot_id

# def get_overshoot(angles): 
#     return max(abs(np.diff(angles))) if len(angles) > 1 else 0.0

# def get_settling_time(angles, steps): 
#     settling_time = None
#     for i in range(1, len(angles)):
#         if abs(angles[i] - angles[0]) < 0.1:
#             settling_time = steps[i]
#             break
#     return settling_time if settling_time else steps[-1] if steps else 0

# def evaluate_controller(env, controller, episodes, max_steps, use_gui=True, num_manual_disturbances=10):
#     rewards, energy_list, smoothness_list, overshoots, settling_times = [], [], [], [], []
#     fall_count = 0

#     # Launch GUI and enable mouse picking if requested
#     if use_gui:
#         if p.getConnectionInfo()['isConnected'] == 0:
#             p.connect(p.GUI)
#         p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
#         p.configureDebugVisualizer(p.COV_ENABLE_GUI, 1)
#     else:
#         p.connect(p.DIRECT)

#     robot_id = load_robot()

#     for ep in range(episodes):
#         obs, _ = env.reset(seed=ep)
#         total_reward = 0
#         torques, angles, steps = [], [], []

#         # Calculate disturbance steps (evenly spaced)
#         disturbance_steps = np.linspace(0, max_steps-1, num_manual_disturbances+2, dtype=int)[1:-1]
#         disturbance_idx = 0

#         for step in range(max_steps):
#             # Pause for manual disturbance at specified steps
#             if (use_gui and disturbance_idx < len(disturbance_steps) and step == disturbance_steps[disturbance_idx]):
#                 print(f"\n[Manual Disturbance {disturbance_idx+1}/{num_manual_disturbances}]")
#                 print("Pause: Use mouse in the PyBullet GUI to disturb the robot. Press Enter to continue...")
#                 input()
#                 disturbance_idx += 1

#             if use_gui:
#                 time.sleep(1.0/60.0)  # Slow down for human interaction

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
#             angles.append(obs[0])  # assuming obs[0] is the pole angle
#             steps.append(step)

#             if done:
#                 if abs(obs[0]) > 0.8: fall_count += 1
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

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     parser.add_argument('--no_gui', action='store_true', help="Disable GUI and mouse interaction")
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
#     env = gymnasium.make("AssistiveWalkerContinuousEnv-v0", render_mode="True")
#     env.action_space.seed(getattr(config, "seed", 0))
#     np.random.seed(getattr(config, "seed", 0))
#     random.seed(getattr(config, "seed", 0))

#     # Load model and controller
#     model = PPO.load(args.model_path)
#     controller = DRLController(model, "PPO", env.action_space)

#     # Evaluate with GUI and mouse interaction unless --no_gui is specified
#     logger.info("Evaluating AssistiveWalkerContinuousEnv with PPO and manual mouse disturbances...")
#     rewards, overshoots, settling_times, energy_list, smoothness_list, metrics = evaluate_controller(
#         env, controller, getattr(config, "eval_episodes", 10), getattr(config, "max_steps", 500),
#         use_gui=not args.no_gui, num_manual_disturbances=10
#     )
#     logger.info(f"Results: {metrics}")

#     # Log to WandB after evaluation
#     if args.use_wandb:
#         log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config)
#         wandb.finish()

#     env.close()
