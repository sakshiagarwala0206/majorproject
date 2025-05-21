import os
import sys
import time
import numpy as np
import pandas as pd
import yaml
import wandb
from gymnasium.envs.registration import register

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environments.walker_1 import AssistiveWalkerContinuousEnv

# --- Load config from YAML ---
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()
with open(args.config, 'r') as f:
    config = yaml.safe_load(f)

# --- WandB setup ---
wandb.init(
    project=config["wandb_project"],
    entity=config.get("wandb_entity", None),
    config=config,
    mode="online"
)

# --- PID Controller ---
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

register(
    id="AssistiveWalkerContinuousEnv-v0",
    entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
    max_episode_steps=1000,
)

def get_disk_usage():
    import psutil
    return round(psutil.disk_usage('.').used / (1024 ** 2), 2)  # In MB

max_episodes = config["max_episodes"]
settling_threshold = config["settling_threshold"]
dt = config["dt"]

env = AssistiveWalkerContinuousEnv(render_mode=True)
pid = PIDController(Kp=config["Kp"], Ki=config["Ki"], Kd=config["Kd"])

for episode in range(max_episodes):
    obs, _ = env.reset()
    pid.reset()
    done = False
    total_reward = 0
    step = 0
    angle_history = []
    action_history = []
    smoothness_metric = []
    max_angle = 0
    settled = False
    settling_time = 0
    t0 = time.time()

    # For IMU logging
    imu_keys = [
        "imu_roll", "imu_pitch", "imu_yaw",
        "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
        "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"
    ]
    imu_episode = {k: [] for k in imu_keys}
    steps_list = []

    while not done:
        pole_angle = obs[0]
        angle_error = pole_angle
        torque_command = pid.compute(angle_error, dt)
        action = np.array([torque_command, -torque_command], dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        angle = obs[0]
        angle_history.append(angle)
        action_history.append(torque_command)
        step += 1
        steps_list.append(step)

        max_angle = max(max_angle, abs(angle))

        if not settled and abs(angle) < settling_threshold:
            settling_time = time.time() - t0
            settled = True

        if len(action_history) > 5:
            var = np.var(action_history[-5:])
            smoothness_metric.append(var)

        # Log IMU data per step (to wandb and to list for plotting)
        if len(obs) >= 16:
            imu_dict = {
                "imu_roll": obs[7],
                "imu_pitch": obs[8],
                "imu_yaw": obs[9],
                "imu_ang_vel_x": obs[10],
                "imu_ang_vel_y": obs[11],
                "imu_ang_vel_z": obs[12],
                "imu_lin_acc_x": obs[13],
                "imu_lin_acc_y": obs[14],
                "imu_lin_acc_z": obs[15],
                "step": step,
                "episode": episode + 1
            }
            wandb.log(imu_dict)
            for k in imu_keys:
                imu_episode[k].append(imu_dict[k])

    # --- Log episode metrics to wandb ---
    wandb.log({
        "episode": episode,
        "episode_reward": total_reward,
        "overshoot": max_angle,
        "settling_time": settling_time,
        "smoothness": np.mean(smoothness_metric) if smoothness_metric else 0.0,
        "episode_length": step,
        "disk_usage_MB": get_disk_usage(),
        "global_step": episode
    })

    print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {max_angle:.3f}, Settling: {settling_time:.2f}s")

    # --- Log IMU time series for this episode as wandb line plots ---
    imu_table = wandb.Table(
        data=list(zip(steps_list, *[imu_episode[k] for k in imu_keys])),
        columns=["step"] + imu_keys
    )
    for k in imu_keys:
        wandb.log({
            f"IMU/{k}_episode_{episode+1}": wandb.plot.line_series(
                xs=steps_list,
                ys=[imu_episode[k]],
                keys=[k],
                title=f"{k} (Episode {episode+1})",
                xname="step"
            )
        })

env.close()