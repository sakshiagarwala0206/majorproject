import time
import gymnasium as gym
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise

from environments.cartpole_env import InvertedPendulumEnvGymnasium  # 🔄 Your custom environment class

# 🎯 Load the trained model
model = DDPG.load("ddpg_inverted_pendulum")

# 🛠️ Create the environment
env = InvertedPendulumEnvGymnasium(render_mode="human")
obs, _ = env.reset()

# ➕ Add action noise for inference (optional, helps prevent getting stuck)
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.05 * np.ones(env.action_space.shape))

# 🎬 Run inference loop
step = 0
episode_reward = 0
steps_in_balance = 0

try:
    while True:
        # 🧠 Get deterministic action from model
        action, _ = model.predict(obs, deterministic=True)

        # 🎲 Add small noise
        noisy_action = np.clip(action + action_noise(), env.action_space.low, env.action_space.high)

        # 🚶 Step environment
        obs, reward, terminated, truncated, info = env.step(noisy_action)
        episode_reward += reward

        # 📏 Track balance based on pole angle
        pole_angle = obs[2]  # Assuming 3rd obs is pole angle
        if abs(pole_angle) < 0.05:
            steps_in_balance += 1
        else:
            steps_in_balance = 0

        # ✅ Terminate early if balanced long enough
        if steps_in_balance > 50:
            print(f"\n🎯 Balanced for {steps_in_balance} steps. Ending episode early.")
            terminated = True

        # ⏱️ 60 FPS simulation
        time.sleep(1.0 / 60.0)

        # 📢 Print status every 100 steps
        if step % 100 == 0:
            print(f"Step {step} | Reward: {reward:.2f} | Action: {noisy_action} | Pole Angle: {pole_angle:.3f}")

        step += 1

        if terminated or truncated:
            print(f"\n🏁 Episode finished | Total Reward: {episode_reward:.2f} | Steps: {step}\n")
            obs, _ = env.reset()
            step = 0
            episode_reward = 0
            steps_in_balance = 0

finally:
    env.close()
