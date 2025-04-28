import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import DDPG  # Use DDPG instead of PPO
import os
import sys

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import the custom CartPoleEnv class
from trash_test_code.cartpole_env import CartPoleEnv
from gymnasium.envs.registration import register
from stable_baselines3.common.logger import configure

# ✅ Register your custom Gymnasium environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole_env:CartPoleEnv',
)

# Load the trained model
model_path = "ddpg_cartpole.zip"  # Update path if needed
model = DDPG.load(model_path)

# Create your custom environment
env = CartPoleEnv()  # This should be compatible with Gymnasium

# Define number of test episodes
n_test_episodes = 1000
episode_rewards = []
episode_lengths = []

# Run test episodes
for episode in range(n_test_episodes):
    obs, _ = env.reset()  # Unpack the observation tuple (Gymnasium returns (obs, info))
    terminated = False
    truncated = False
    total_reward = 0
    episode_length = 0

    while not (terminated or truncated):
        # Predict action using trained model
        action, _states = model.predict(obs, deterministic=True)
        
        # Interact with the environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        total_reward += reward
        episode_length += 1

    # Store results
    episode_rewards.append(total_reward)
    episode_lengths.append(episode_length)

# Summarize results
average_reward = np.mean(episode_rewards)
average_length = np.mean(episode_lengths)

print(f"\nTest Results over {n_test_episodes} episodes:")
print(f"Average Episode Reward: {average_reward:.2f}")
print(f"Average Episode Length: {average_length:.2f}")

# Visualization
plt.figure(figsize=(12, 6))

# Episode Reward
plt.subplot(1, 2, 1)
plt.plot(range(1, n_test_episodes + 1), episode_rewards, marker='o', label="Episode Reward")
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Episode Reward During Testing")
plt.grid(True)

# Episode Length
plt.subplot(1, 2, 2)
plt.plot(range(1, n_test_episodes + 1), episode_lengths, marker='o', label="Episode Length")
plt.xlabel("Episode")
plt.ylabel("Length")
plt.title("Episode Length During Testing")
plt.grid(True)

plt.tight_layout()
plt.show()
