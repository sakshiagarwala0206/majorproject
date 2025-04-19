from stable_baselines3 import DDPG
from stable_baselines3.common.evaluation import evaluate_policy
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import logging

# Load trained model
model = DDPG.load("ddpg_cartpole_inference.zip")  # Update path if needed
from gymnasium.envs.registration import register

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Now you can import the environment
from environments.cartpole_env import CartPoleEnv  # Import the CartPoleEnv class



# ✅ Register your environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole_env:CartPoleEnv',  # Correct the entry point
)

# Run a single evaluation episode to log angle
obs, _ = env.reset()
angles = []
rewards = []
done = False

while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    pole_angle = obs[2]  # Assuming 3rd obs is pole angle in radians
    angles.append(pole_angle)
    rewards.append(reward)

# Plot angle over time
plt.plot(np.rad2deg(angles))  # convert to degrees for intuition
plt.title("Pole Angle over Time")
plt.xlabel("Time step")
plt.ylabel("Angle (degrees)")
plt.grid(True)
plt.show()

# Compute Overshoot (max deviation from 0)
overshoot_deg = np.max(np.abs(np.rad2deg(angles)))
print("Overshoot:", overshoot_deg, "degrees")

# Compute Settle Time (within ±2° ~ 0.0349 rad)
threshold = np.deg2rad(2)
for i in range(len(angles)):
    if all(abs(a) < threshold for a in angles[i:]):
        settle_time = i
        break
else:
    settle_time = len(angles)

print("Settle Time:", settle_time, "steps")

# Evaluate over multiple episodes to assess robustness
mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
print("Mean Reward:", mean_reward)
print("Std Reward:", std_reward)
