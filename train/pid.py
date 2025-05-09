import gym
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import time
import psutil
import sys
from gymnasium.envs.registration import register
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environments.cartpole import CartPoleContinuousEnv  # your custom env

# PID Controller class
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
    id="CartPoleCont",
    entry_point="environments.cartpole:CartPoleContiniousEnv",
    max_episode_steps=500,  # same as Gym CartPole
)
# Initialize environment
# env = gym.make("CartPoleCont")  # Ensure your URDF env is registered

env = CartPoleContinuousEnv(render_mode=True)
pid = PIDController(Kp=8.2, Ki=0.4, Kd=2.5)

# Data logging dictionary
log_data = {
    "episode": [],
    "episode_reward": [],
    "overshoot": [],
    "settling_time": [],
    "smoothness": [],
    "episode_length": [],
    "disk_usage_MB": [],
    "global_step": []
}

# Settings
max_episodes = 30
settling_threshold = 0.05  # Define acceptable range for settling

def get_disk_usage():
    return round(psutil.disk_usage('.').used / (1024 ** 2), 2)  # In MB

def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
    for t, a in zip(times, angles):
        if abs(a - target_angle) <= tolerance:
            return float(t)
    return float(max(times))

# Run PID simulation
for episode in range(max_episodes):
    obs = env.reset()
    pid.reset()
    done = False
    total_reward = 0
    step = 0
    dt = 0.05  # time delta
    angle_history = []
    action_history = []
    smoothness_metric = []
    max_angle = 0
    settled = False
    settling_time = 0
    t0 = time.time()

    while not done:
        angle_error = obs[0]  # Adjust if needed based on your env
        action = pid.compute(angle_error, dt)
        # obs, reward, done, info = env.step(action)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        total_reward += reward
        angle = obs[0]
        angle_history.append(angle)
        action_history.append(action)
        step += 1

        # Overshoot
        max_angle = max(max_angle, abs(angle))
    # times= np.arange(len(angle_history)) * dt
    # # settling_time = get_settling_time(times, angle_history, target_angle=0.0, tolerance=0.05)

            # Settling time
        if not settled and abs(angle) < settling_threshold:
                settling_time = time.time() - t0
                settled = True

        # Smoothness: variance of recent actions
    if len(action_history) > 5:
            var = np.var(action_history[-5:])
            smoothness_metric.append(var)

    # Logging per episode
    log_data["episode"].append(episode)
    log_data["episode_reward"].append(total_reward)
    log_data["overshoot"].append(max_angle)
    log_data["settling_time"].append(settling_time)
    log_data["smoothness"].append(np.mean(smoothness_metric))
    log_data["episode_length"].append(step)
    log_data["disk_usage_MB"].append(get_disk_usage())
    log_data["global_step"].append(episode)

    print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {max_angle:.3f}, Settling: {settling_time:.2f}s")

env.close()

# Save results
df = pd.DataFrame(log_data)
df.to_csv("pid_metrics.csv", index=False)
print("Saved metrics to pid_metrics.csv")

# Plotting
def plot_metric(metric_name):
    plt.figure(figsize=(10, 5))
    plt.plot(df["episode"], df[metric_name], label=f"PID {metric_name}")
    plt.xlabel("Episode")
    plt.ylabel(metric_name.replace('_', ' ').title())
    plt.title(f"{metric_name.replace('_', ' ').title()} Over Episodes (PID)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/pid_{metric_name}.png")
    plt.show()

# Create folder for plots
os.makedirs("plots", exist_ok=True)

# Plot all metrics
for metric in ["episode_reward", "overshoot", "settling_time", "smoothness", "episode_length", "disk_usage_MB"]:
    plot_metric(metric)
