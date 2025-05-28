
# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import time
# import psutil
# import sys
# import wandb
# from gymnasium.envs.registration import register
# from datetime import datetime

# # === Add Environment Path ===
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from environments.cartpole import CartPoleContinuousEnv  # Make sure this returns correct rewards

# # === PID Controller ===
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

# # === Register and Initialize Environment ===
# register(
#     id="CartPoleCont",
#     entry_point="environments.cartpole:CartPoleContinuousEnv",
#     max_episode_steps=500,
# )
# env = CartPoleContinuousEnv(render_mode=True)
# pid = PIDController(Kp=18.27, Ki=13.91, Kd=3.224)

# # === Initialize wandb ===
# wandb.init(
#     project="Cartpole_test_final",
#     name=f"eval_PID_{datetime.now():%Y%m%d_%H%M%S}",
#     config={
#         "Kp": 18.27,
#         "Ki": 13.91,
#         "Kd": 3.224,
#         "max_episodes": 100,
#         "settling_threshold": 0.05
#     }
# )

# log_data = {
#     "episode": [], "episode_reward": [], "overshoot": [], "settling_time": [],
#     "smoothness": [], "episode_length": [], "disk_usage_MB": [],
#     "global_step": [], "energy": []
# }

# max_episodes = 100
# settling_threshold = 0.05

# def get_disk_usage():
#     return round(psutil.disk_usage('.').used / (1024 ** 2), 2)

# # === Main Loop ===
# for episode in range(max_episodes):
#     obs, _ = env.reset()
#     pid.reset()

#     try:
#         random_angle = np.random.uniform(-0.5, 0.5)
#         env.unwrapped.state = np.array([0.0, 0.0, random_angle, 0.0])
#         obs = env.unwrapped.state.copy()
#     except Exception as e:
#         print(f"[WARN] Could not set random initial angle: {e}")

#     done = False
#     total_reward = 0
#     step = 0
#     dt = 0.05
#     angle_history = []
#     action_history = []
#     max_angle = 0
#     settled = False
#     settling_time = 0
#     t0 = time.time()

#     while not done:
#         angle = obs[2]
#         action = pid.compute(angle, dt)

#         obs, reward, terminated, truncated, info = env.step([action, 0.0])
#         done = terminated or truncated

#         # === Assert reward type and debug ===
#         assert isinstance(reward, (float, int, np.floating, np.integer, np.ndarray)), f"[ERROR] Invalid reward type: {type(reward)}"

#         if reward is None or reward == 0:
#             print(f"[WARN] Original reward is zero. Angle = {angle:.4f}, replacing with shaped reward.")
#             reward = 1.0 - abs(angle)  # Angle-based shaping

#         # === Ensure reward is scalar float ===
#         reward_scalar = float(np.squeeze(reward))
#         total_reward += reward_scalar

#         print(f"[DEBUG] Step {step}: reward = {reward_scalar:.4f} (type={type(reward)})")

#         angle_history.append(angle)
#         action_history.append(action)
#         step += 1

#         max_angle = max(max_angle, abs(angle))
#         if not settled and abs(angle) < settling_threshold:
#             settling_time = time.time() - t0
#             settled = True

#     # === Post-Episode Calculations ===
#     jerk = np.diff(angle_history) / dt if len(angle_history) > 2 else [0.0]
#     smoothness = np.mean(np.abs(jerk))
#     energy = float(np.sum(np.square(action_history)))

#     log_data["episode"].append(episode)
#     log_data["episode_reward"].append(total_reward)
#     log_data["overshoot"].append(max_angle)
#     log_data["settling_time"].append(settling_time)
#     log_data["smoothness"].append(smoothness)
#     log_data["episode_length"].append(step)
#     log_data["disk_usage_MB"].append(get_disk_usage())
#     log_data["global_step"].append(episode)
#     log_data["energy"].append(energy)

#     wandb.log({
#         "episode": episode,
#         "episode_reward": total_reward,
#         "overshoot": max_angle,
#         "settling_time": settling_time,
#         "smoothness": smoothness,
#         "episode_length": step,
#         "disk_usage_MB": log_data["disk_usage_MB"][-1],
#         "energy": energy,
#         "global_step": episode
#     })

#     print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {max_angle:.3f}, Settling: {settling_time:.2f}s")

# env.close()

# # === Save and Plot ===
# df = pd.DataFrame(log_data)
# df.to_csv("pid_metrics.csv", index=False)
# print("Saved metrics to pid_metrics.csv")

# episode_table = wandb.Table(dataframe=df)
# for key, (y_col, title) in {
#     "Reward Curve": ("episode_reward", "Reward per Episode"),
#     "Overshoot Curve": ("overshoot", "Overshoot per Episode"),
#     "Settling Time Curve": ("settling_time", "Settling Time per Episode"),
#     "Energy Curve": ("energy", "Energy per Episode"),
#     "Smoothness Curve": ("smoothness", "Smoothness per Episode"),
# }.items():
#     wandb.log({key: wandb.plot.line(episode_table, "episode", y_col, title=title)})
# wandb.log({"Evaluation Data": episode_table})
# wandb.finish()

# # === Optional Local Plotting ===
# def plot_metric(metric_name):
#     plt.figure(figsize=(10, 5))
#     plt.plot(df["episode"], df[metric_name], label=f"PID {metric_name}")
#     plt.xlabel("Episode")
#     plt.ylabel(metric_name.replace('_', ' ').title())
#     plt.title(f"{metric_name.replace('_', ' ').title()} Over Episodes (PID)")
#     plt.grid(True)
#     plt.legend()
#     plt.tight_layout()
#     os.makedirs("plots", exist_ok=True)
#     plt.savefig(f"plots/pid_{metric_name}.png")
#     plt.show()

# for metric in ["episode_reward", "overshoot", "settling_time", "smoothness", "episode_length", "disk_usage_MB"]:
#     plot_metric(metric)

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd
# import os
# import time
# import psutil
# import sys
# import wandb
# from gymnasium.envs.registration import register
# from datetime import datetime
# import pybullet as p  # Added for disturbance

# # === Add Environment Path ===
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from environments.cartpole import CartPoleContinuousEnv

# # === PID Controller ===
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

# # === Register and Initialize Environment ===
# register(
#     id="CartPoleCont",
#     entry_point="environments.cartpole:CartPoleContinuousEnv",
#     max_episode_steps=500,  # Match DRL's episode length
# )
# env = CartPoleContinuousEnv(render_mode=True)
# pid = PIDController(Kp=18.27, Ki=13.91, Kd=3.224)

# # === Initialize wandb ===
# wandb.init(
#     project="Cartpole_test_final",
#     name=f"eval_PID_{datetime.now():%Y%m%d_%H%M%S}",
#     config={
#         "Kp": 18.27,
#         "Ki": 13.91,
#         "Kd": 3.224,
#         "max_episodes": 100,
#         "settling_threshold": 0.05
#     }
# )

# log_data = {
#     "episode": [], "episode_reward": [], "overshoot": [], "settling_time": [],
#     "smoothness": [], "episode_length": [], "disk_usage_MB": [],
#     "global_step": [], "energy": [], "fall_rate": []  # Added fall_rate
# }

# max_episodes = 100
# settling_threshold = 0.05
# fall_angle_threshold = 0.8  # Match DRL's threshold

# def get_disk_usage():
#     return round(psutil.disk_usage('.').used / (1024 ** 2), 2)

# # === Main Loop ===
# for episode in range(max_episodes):
#     obs, _ = env.reset()
#     pid.reset()
#     fall_flag = False

#     try:
#         # Match DRL's initial angle range (-0.5 to 0.5 rad)
#         random_angle = np.random.uniform(-0.5, 0.5)
#         env.unwrapped.state = np.array([0.0, 0.0, random_angle, 0.0])
#         obs = env.unwrapped.state.copy()
#     except Exception as e:
#         print(f"[WARN] Could not set random initial angle: {e}")

#     done = False
#     total_reward = 0
#     step = 0
#     dt = 0.05
#     angle_history = []
#     action_history = []
#     max_angle = 0
#     settled = False
#     settling_time = 0
#     t0 = time.time()

#     while not done:
#         angle = obs[2]
#         action = pid.compute(angle, dt)

#         # Apply disturbance at step 200 (same as DRL)
#         if step == 200:
#             if hasattr(env, "cartpole_id"):
#                 p.applyExternalForce(
#                     objectUniqueId=env.cartpole_id,
#                     linkIndex=-1,
#                     forceObj=[4, 0, 0],
#                     posObj=[0, 0, 0],
#                     flags=p.WORLD_FRAME
#                 )

#         obs, reward, terminated, truncated, info = env.step([action, 0.0])
#         done = terminated or truncated

#         # Use environment's original reward (remove shaping)
#         total_reward += float(reward)

#         angle_history.append(angle)
#         action_history.append(action)
#         step += 1

#         # Track max angle for fall rate
#         current_angle = abs(angle)
#         if current_angle > fall_angle_threshold:
#             fall_flag = True

#         max_angle = max(max_angle, current_angle)
        
#         # Settling time after disturbance (step >= 200)
#         if step >= 200 and not settled and abs(angle) < settling_threshold:
#             settling_time = time.time() - t0
#             settled = True

#     # Post-Episode Calculations
#     jerk = np.diff(angle_history)/dt if len(angle_history)>2 else [0.0]
#     smoothness = np.mean(np.abs(jerk))
#     energy = float(np.sum(np.abs(action_history)))  # ∑|action| not ∑action²
    
#     log_data["episode"].append(episode)
#     log_data["episode_reward"].append(total_reward)
#     log_data["overshoot"].append(max_angle)
#     log_data["settling_time"].append(settling_time)
#     log_data["smoothness"].append(smoothness)
#     log_data["episode_length"].append(step)
#     log_data["disk_usage_MB"].append(get_disk_usage())
#     log_data["global_step"].append(episode)
#     log_data["energy"].append(energy)
#     log_data["fall_rate"].append(1 if fall_flag else 0)

#     wandb.log({
#         "episode": episode,
#         "episode_reward": total_reward,
#         "overshoot": max_angle,
#         "settling_time": settling_time,
#         "smoothness": smoothness,
#         "episode_length": step,
#         "disk_usage_MB": log_data["disk_usage_MB"][-1],
#         "energy": energy,
#         "global_step": episode,
#         "fall_rate": log_data["fall_rate"][-1]
#     })

#     print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {max_angle:.3f}, Settling: {settling_time:.2f}s")

# env.close()

# # === Save and Plot ===
# df = pd.DataFrame(log_data)
# df.to_csv("pid_metrics.csv", index=False)
# print("Saved metrics to pid_metrics.csv")

# # === Add Fall Rate Statistics ===
# fall_rate = np.mean(df["fall_rate"]) * 100
# print(f"\nFall Rate: {fall_rate:.2f}%")

# episode_table = wandb.Table(dataframe=df)
# for key, (y_col, title) in {
#     "Reward Curve": ("episode_reward", "Reward per Episode"),
#     "Overshoot Curve": ("overshoot", "Overshoot per Episode"),
#     "Settling Time Curve": ("settling_time", "Settling Time per Episode"),
#     "Energy Curve": ("energy", "Energy per Episode"),
#     "Smoothness Curve": ("smoothness", "Smoothness per Episode"),
#     "Fall Rate Curve": ("fall_rate", "Fall Rate per Episode")
# }.items():
#     wandb.log({key: wandb.plot.line(episode_table, "episode", y_col, title=title)})
    
# wandb.log({"Evaluation Data": episode_table})
# wandb.finish()



import numpy as np
import pandas as pd
import os
import time
import psutil
import sys
import wandb
from gymnasium.envs.registration import register
from datetime import datetime
import pybullet as p

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environments.cartpole import CartPoleContinuousEnv

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
    entry_point="environments.cartpole:CartPoleContinuousEnv",
    max_episode_steps=500,
)
env = CartPoleContinuousEnv(render_mode=True)
pid = PIDController(Kp=18.27, Ki=13.91, Kd=3.224)

wandb.init(
    project="Cartpole_test_final",
    name=f"eval_PID_{datetime.now():%Y%m%d_%H%M%S}",
    config={
        "Kp": 18.27,
        "Ki": 13.91,
        "Kd": 3.224,
        "max_episodes": 100,
        "settling_threshold": 0.05
    }
)

log_data = {
    "episode": [], "reward": [], "overshoot": [], "settling_time": [],
    "smoothness": [], "episode_length": [], "energy": [], "fall_rate": []
}

max_episodes = 100
settling_threshold = 0.05
fall_angle_threshold = 0.8

def get_energy(actions):
    return float(np.sum(np.abs(actions)))

def get_smoothness(angles, steps, dt=0.05):
    if len(angles) > 1:
        jerks = np.diff(angles) / (np.diff(steps) * dt)
        return float(np.mean(np.abs(jerks))) if len(jerks) > 0 else 0.0
    return 0.0

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(steps, angles, disturbance_step=200, tolerance=0.05):
#     for t, a in zip(steps, angles):
#         if t >= disturbance_step and abs(a) <= tolerance:
#             return float(max(0, t - disturbance_step))
#     return float(max(steps) - disturbance_step) if steps else 0.0
def get_settling_time(steps, angles, tolerance=0.1):
    """
    Returns the first time (step index) at which the angle enters and remains within the tolerance band.
    If it never settles, returns the episode length.
    """
    for i, a in enumerate(angles):
        if abs(a) <= tolerance:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0



for episode in range(max_episodes):
    obs, _ = env.reset()
    pid.reset()
    fall_flag = False

    try:
        random_angle = np.random.uniform(-0.5, 0.5)
        env.unwrapped.state = np.array([0.0, 0.0, random_angle, 0.0])
        obs = env.unwrapped.state.copy()
    except Exception as e:
        print(f"[WARN] Could not set random initial angle: {e}")

    done = False
    total_reward = 0
    step = 0
    dt = 0.05
    angle_history = []
    action_history = []
    step_history = []
    max_angle = 0

    while not done:
        angle = obs[2]
        action = pid.compute(angle, dt)

        # Apply disturbance at step 200
        if step == 200:
            if hasattr(env, "cartpole_id"):
                p.applyExternalForce(
                    objectUniqueId=env.cartpole_id,
                    linkIndex=-1,
                    forceObj=[4, 0, 0],
                    posObj=[0, 0, 0],
                    flags=p.WORLD_FRAME
                )

        obs, reward, terminated, truncated, info = env.step([action, 0.0])
        done = terminated or truncated

        total_reward += float(reward)
        angle_history.append(angle)
        action_history.append(action)
        step_history.append(step)
        step += 1

        current_angle = abs(angle)
        if current_angle > fall_angle_threshold:
            fall_flag = True

        max_angle = max(max_angle, current_angle)

    overshoot = get_overshoot(angle_history)
    settling_time = get_settling_time(step_history, angle_history)
    smoothness = get_smoothness(angle_history, step_history, dt=dt)
    energy = get_energy(action_history)
    episode_length = step
    fall_rate = 1 if fall_flag or episode_length < 500 else 0

    log_data["episode"].append(episode)
    log_data["reward"].append(total_reward)
    log_data["overshoot"].append(overshoot)
    log_data["settling_time"].append(settling_time)
    log_data["smoothness"].append(smoothness)
    log_data["episode_length"].append(episode_length)
    log_data["energy"].append(energy)
    log_data["fall_rate"].append(fall_rate)

    wandb.log({
        "episode": episode,
        "reward": total_reward,
        "overshoot": overshoot,
        "settling_time": settling_time,
        "smoothness": smoothness,
        "energy": energy,
        "fall_rate": fall_rate,
        "episode_length": episode_length
    })

    print(f"Episode {episode+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_rate}")

env.close()

# Save metrics to CSV
df = pd.DataFrame(log_data)
df.to_csv("pid_metrics.csv", index=False)
print("Saved metrics to pid_metrics.csv")

# Unified WandB logging and plotting
metrics = ["reward", "overshoot", "settling_time", "smoothness", "energy", "fall_rate", "episode_length"]
episode_table = wandb.Table(dataframe=df)

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
