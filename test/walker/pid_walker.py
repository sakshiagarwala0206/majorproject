import argparse
import os
import sys
import time
import numpy as np
import yaml
import wandb
import random
import pybullet as p
from gymnasium.envs.registration import register

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environments.walker import AssistiveWalkerContinuousEnv

class PIDController:
    def __init__(self, Kp, Ki, Kd):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def reset(self):
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt if dt > 0 else 0
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error
        return output

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

def get_settling_time(steps, angles, tolerance=0.1):
    for i, a in enumerate(angles):
        if abs(a) <= tolerance:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0

def load_robot():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
    return robot_id

def evaluate_pid(env, pid, config):
    max_episodes = config.max_episodes
    dt = config.dt
    settling_threshold = config.settling_threshold

    log_data = {"episode": [], "reward": [], "overshoot": [], "settling_time": [],
                "smoothness": [], "energy": [], "fall_rate": [], "episode_length": []}
    robot_id = load_robot()

    for episode in range(max_episodes):
        obs, _ = env.reset(seed=episode)
        pid.reset()
        total_reward = 0
        step = 0
        angle_history = []
        action_history = []
        steps_list = []
        fall_flag = False

        for step in range(config.max_steps):
            if step == 100:
                base_mass = p.getDynamicsInfo(robot_id, -1)[0]
                p.changeDynamics(robot_id, -1, mass=base_mass + 20)
            if step == 200:
                p.applyExternalForce(robot_id, -1, [4, 0, 0], [0, 0, 0], p.WORLD_FRAME)

            angle_error = obs[2]
            torque_command = pid.compute(angle_error, dt)
            action = np.array([torque_command, -torque_command], dtype=np.float32)
            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            angle = obs[2]
            angle_history.append(angle)
            action_history.append(torque_command)
            steps_list.append(step)
            total_reward += reward

            if done:
                if abs(angle) > 0.8 or step < config.max_steps - 1:
                    fall_flag = True
                break

        overshoot = get_overshoot(angle_history)
        settling_time = get_settling_time(steps_list, angle_history, tolerance=settling_threshold)
        smoothness = get_smoothness(angle_history, steps_list, dt=dt)
        energy = get_energy(action_history)

        log_data["episode"].append(episode)
        log_data["reward"].append(total_reward)
        log_data["overshoot"].append(overshoot)
        log_data["settling_time"].append(settling_time)
        log_data["smoothness"].append(smoothness)
        log_data["energy"].append(energy)
        log_data["fall_rate"].append(1 if fall_flag else 0)
        log_data["episode_length"].append(step + 1)

        print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, "
              f"Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_flag}")

    import pandas as pd
    df = pd.DataFrame(log_data)
    df.to_csv("walker_pid_metrics.csv", index=False)
    print("Saved metrics to walker_pid_metrics.csv")

    summary = {f"{m}_mean": float(df[m].mean()) for m in df.columns if m != "episode"}
    summary.update({f"{m}_std": float(df[m].std()) for m in df.columns if m != "episode"})
    summary.update({f"{m}_median": float(df[m].median()) for m in df.columns if m != "episode"})

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
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config_dict = yaml.safe_load(f)
    config = argparse.Namespace(**config_dict)

    if args.use_wandb:
        wandb.init(
            project="assistive-walker-test",
            config=config_dict,
            name=f"eval_PID_{time.strftime('%Y%m%d_%H%M%S')}"
        )

    register(
        id="WalkerBalanceContinuousEnv-v0",
        entry_point="environments.walker:AssistiveWalkerContinuousEnv",
        max_episode_steps=1000
    )

    env = AssistiveWalkerContinuousEnv(render_mode=True)
    env.action_space.seed(getattr(config, "seed", 0))
    np.random.seed(getattr(config, "seed", 0))
    random.seed(getattr(config, "seed", 0))

    pid = PIDController(Kp=config.Kp, Ki=config.Ki, Kd=config.Kd)
    df, summary = evaluate_pid(env, pid, config)

    if args.use_wandb:
        log_to_wandb(df, summary)
        wandb.finish()

    env.close()















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
# import pybullet as p

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from environments.walker import AssistiveWalkerContinuousEnv
# from controllers.drl_controller import DRLController
# from train.utils.logger import setup_logger
# from stable_baselines3 import PPO

# logger = setup_logger()

# register(
#     id="WalkerBalanceContinuousEnv-v0",
#     entry_point="environments.walker:AssistiveWalkerContinuousEnv"  ,
#     max_episode_steps=1000,
# )

# def load_config(path):
#     with open(path, 'r') as f:
#         return SimpleNamespace(**yaml.safe_load(f))

# def load_robot():
#     project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#     urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#     if not os.path.exists(urdf_path):
#         raise FileNotFoundError(f"URDF file not found at {urdf_path}")
#     robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
#     return robot_id

# # --- Metric Functions (same as CartPole) ---
# def get_energy(forces):
#     flat = []
#     for v in forces:
#         try:
#             vals = list(v)
#         except TypeError:
#             vals = [v]
#         flat.extend([abs(x) for x in vals])
#     return float(sum(flat))

# def get_smoothness(angles, steps, dt=1.0):
#     if len(angles) > 1 and len(steps) > 1:
#         jerks = np.diff(angles) / (np.diff(steps) * dt)
#         if len(jerks) > 0 and not np.any(np.isnan(jerks)):
#             return float(np.mean(np.abs(jerks)))
#     return 0.0

# def get_overshoot(angles, target_angle=0.0):
#     return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(steps, angles, tolerance=0.1):
#     for i, a in enumerate(angles):
#         if abs(a) <= tolerance:
#             return float(steps[i])
#     return float(steps[-1]) if steps else 0.0

# def evaluate_controller(env, controller, episodes, max_steps):
#     log_data = {
#         "episode": [], "reward": [], "overshoot": [], "settling_time": [],
#         "smoothness": [], "energy": [], "fall_rate": [], "episode_length": []
#     }
#     fall_count = 0
#     robot_id = load_robot()

#     for ep in range(episodes):
#         obs, _ = env.reset(seed=ep)
#         total_reward = 0.0
#         episode_forces = []
#         episode_angles = []
#         episode_steps = []
#         done = False
#         fall_flag = False

#         for step in range(max_steps):
#             # Disturbances
#             if step == 100:
#                 base_mass = p.getDynamicsInfo(robot_id, -1)[0]
#                 p.changeDynamics(robot_id, -1, mass=base_mass + 20)
#             if step == 200:
#                 p.applyExternalForce(
#                     objectUniqueId=robot_id, linkIndex=-1,
#                     forceObj=[4, 0, 0], posObj=[0, 0, 0], flags=p.WORLD_FRAME
#                 )

#             action, _ = controller.model.predict(obs, deterministic=True)
#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             # For walker, use e.g. obs[2] as the main angle (adapt if needed)
#             episode_angles.append(obs[2])
#             episode_forces.append(action)
#             episode_steps.append(step)
#             total_reward += reward

#             if done:
#                 if abs(obs[2]) > 0.8 or step < max_steps - 1:
#                     fall_flag = True
#                     fall_count += 1
#                 break

#         overshoot = get_overshoot(episode_angles)
#         settling_time = get_settling_time(episode_steps, episode_angles)
#         smoothness = get_smoothness(episode_angles, episode_steps, dt=1.0)
#         energy = get_energy(episode_forces)
#         episode_length = step + 1
#         fall_rate = 1 if fall_flag else 0

#         log_data["episode"].append(ep)
#         log_data["reward"].append(total_reward)
#         log_data["overshoot"].append(overshoot)
#         log_data["settling_time"].append(settling_time)
#         log_data["smoothness"].append(smoothness)
#         log_data["energy"].append(energy)
#         log_data["fall_rate"].append(fall_rate)
#         log_data["episode_length"].append(episode_length)

#         print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_rate}")
#     import pandas as pd
#     # Aggregate metrics
#     metrics = ["reward", "overshoot", "settling_time", "smoothness", "energy", "fall_rate", "episode_length"]
#     df = pd.DataFrame(log_data)
#     df.to_csv("walker_ppo_metrics.csv", index=False)
#     print("Saved metrics to walker_ppo_metrics.csv")

#     summary = {f"{m}_mean": float(df[m].mean()) for m in metrics}
#     summary.update({f"{m}_std": float(df[m].std()) for m in metrics})
#     summary.update({f"{m}_median": float(df[m].median()) for m in metrics})

#     return df, summary

# def log_to_wandb(df, summary):
#     episode_table = wandb.Table(dataframe=df)
#     wandb.log(summary)
#     for metric in df.columns:
#         if metric != "episode":
#             wandb.log({
#                 f"{metric}_curve": wandb.plot.line(
#                     episode_table, "episode", metric, title=f"{metric.replace('_', ' ').title()} Over Episodes"
#                 )
#             })
#     wandb.log({"Evaluation Data": episode_table})

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)

#     if args.use_wandb:
#         wandb.init(
#             project="assistive-walker-test",
#             config=vars(config),
#             name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
#         )

#     env = gym.make("WalkerBalanceContinuousEnv-v0", render_mode="human")
#     env.action_space.seed(getattr(config, "seed", 0))
#     np.random.seed(getattr(config, "seed", 0))
#     random.seed(getattr(config, "seed", 0))

#     model = PPO.load(args.model_path)
#     controller = DRLController(model, "PPO", action_space=env.action_space)

#     df, summary = evaluate_controller(
#         env, controller, getattr(config, "eval_episodes", 100), getattr(config, "max_steps", 500)
#     )

#     if args.use_wandb:
#         log_to_wandb(df, summary)
#         wandb.finish()

#     env.close()






# import os
# import sys
# import time
# import numpy as np
# import pandas as pd
# import yaml
# import wandb
# from gymnasium.envs.registration import register

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from environments.walker_1 import AssistiveWalkerContinuousEnv

# # --- Load config from YAML ---
# import argparse
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True, help='Path to config file')
# args = parser.parse_args()
# with open(args.config, 'r') as f:
#     config = yaml.safe_load(f)

# # --- WandB setup ---
# wandb.init(
#     project=config["wandb_project"],
#     entity=config.get("wandb_entity", None),
#     config=config,
#     mode="online"
# )

# # --- PID Controller ---
# class PIDController:
#     def __init__(self, Kp, Ki, Kd):
#         self.Kp = Kp
#         self.Ki = Ki
#         self.Kd = Kd
#         self.prev_error = 0
#         self.integral = 0

#     def reset(self):
#         self.prev_error = 0
#         self.integral = 0

#     def compute(self, error, dt):
#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt if dt > 0 else 0
#         output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
#         self.prev_error = error
#         return output

# register(
#     id="AssistiveWalkerContinuousEnv-v0",
#     entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
#     max_episode_steps=1000,
# )

# def get_disk_usage():
#     import psutil
#     return round(psutil.disk_usage('.').used / (1024 ** 2), 2)  # In MB

# max_episodes = config["max_episodes"]
# settling_threshold = config["settling_threshold"]
# dt = config["dt"]

# env = AssistiveWalkerContinuousEnv(render_mode=True)
# pid = PIDController(Kp=config["Kp"], Ki=config["Ki"], Kd=config["Kd"])

# for episode in range(max_episodes):
#     obs, _ = env.reset()
#     pid.reset()
#     done = False
#     total_reward = 0
#     step = 0
#     angle_history = []
#     action_history = []
#     smoothness_metric = []
#     max_angle = 0
#     settled = False
#     settling_time = 0
#     t0 = time.time()

#     # For IMU logging
#     imu_keys = [
#         "imu_roll", "imu_pitch", "imu_yaw",
#         "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
#         "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"
#     ]
#     imu_episode = {k: [] for k in imu_keys}
#     steps_list = []

#     while not done:
#         pole_angle = obs[0]
#         angle_error = pole_angle
#         torque_command = pid.compute(angle_error, dt)
#         action = np.array([torque_command, -torque_command], dtype=np.float32)
#         obs, reward, terminated, truncated, info = env.step(action)
#         done = terminated or truncated

#         total_reward += reward
#         angle = obs[0]
#         angle_history.append(angle)
#         action_history.append(torque_command)
#         step += 1
#         steps_list.append(step)

#         max_angle = max(max_angle, abs(angle))

#         if not settled and abs(angle) < settling_threshold:
#             settling_time = time.time() - t0
#             settled = True

#         if len(action_history) > 5:
#             var = np.var(action_history[-5:])
#             smoothness_metric.append(var)

#         # Log IMU data per step (to wandb and to list for plotting)
#         if len(obs) >= 16:
#             imu_dict = {
#                 "imu_roll": obs[7],
#                 "imu_pitch": obs[8],
#                 "imu_yaw": obs[9],
#                 "imu_ang_vel_x": obs[10],
#                 "imu_ang_vel_y": obs[11],
#                 "imu_ang_vel_z": obs[12],
#                 "imu_lin_acc_x": obs[13],
#                 "imu_lin_acc_y": obs[14],
#                 "imu_lin_acc_z": obs[15],
#                 "step": step,
#                 "episode": episode + 1
#             }
#             wandb.log(imu_dict)
#             for k in imu_keys:
#                 imu_episode[k].append(imu_dict[k])

#     # --- Log episode metrics to wandb ---
#     wandb.log({
#         "episode": episode,
#         "episode_reward": total_reward,
#         "overshoot": max_angle,
#         "settling_time": settling_time,
#         "smoothness": np.mean(smoothness_metric) if smoothness_metric else 0.0,
#         "episode_length": step,
#         "disk_usage_MB": get_disk_usage(),
#         "global_step": episode
#     })

#     print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {max_angle:.3f}, Settling: {settling_time:.2f}s")

#     # --- Log IMU time series for this episode as wandb line plots ---
#     imu_table = wandb.Table(
#         data=list(zip(steps_list, *[imu_episode[k] for k in imu_keys])),
#         columns=["step"] + imu_keys
#     )
#     for k in imu_keys:
#         wandb.log({
#             f"IMU/{k}_episode_{episode+1}": wandb.plot.line_series(
#                 xs=steps_list,
#                 ys=[imu_episode[k]],
#                 keys=[k],
#                 title=f"{k} (Episode {episode+1})",
#                 xname="step"
#             )
#         })

# env.close()