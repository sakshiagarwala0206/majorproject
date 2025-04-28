import time
import gym
import torch
import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from trash_test_code.pendulum_env import InvertedPendulumEnv  
import os# import your custom env
from trash_test_code.cartpole_env import CartPoleEnv

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), './'))  # Going to project root
model_path= os.path.join(project_root, "ddpg_inverted_pendulum.zip")

# 🎯 Load the trained model
model = DDPG.load(model_path)

# Create the environment
env = CartPoleEnv(render=True)
obs, _ = env.reset()

# Print action space limits (to ensure actions are within bounds)
print("Action space low:", env.action_space.low)
print("Action space high:", env.action_space.high)

# Add action noise (during inference only for slight movement if stuck)
# Increase noise to allow more exploration
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape))

# Variables to track episode and rewards
step = 0
episode_reward = 0
steps_in_balance = 0  # Track how many steps the pole stays balanced

try:
    while True:
        # Get action from model
        action, _ = model.predict(obs, deterministic=True)

        # Add small noise to avoid freezing (optional)
        noisy_action = np.clip(action + action_noise(), env.action_space.low, env.action_space.high)

        # Step through the environment
        obs, reward, terminated, truncated, info = env.step(noisy_action)
        episode_reward += reward

        # Extract angle for display (assume pole angle is at index 2 in observation)
        pole_angle = obs[2]  # Assuming obs[2] is the pole angle

        # Optionally, penalize for large pole angle deviations
        if abs(pole_angle) > 0.1:  # If pole angle is more than 0.1 radians
            reward -= 0.5  # Apply penalty to the reward

        # Check if pole is within a small angle range (e.g., -0.05 to 0.05)
        if abs(pole_angle) < 0.05:
            steps_in_balance += 1
        else:
            steps_in_balance = 0  # Reset the counter if balance is lost

        # End episode if the pole is balanced for a certain number of steps
        if steps_in_balance > 50:  # Adjust this based on how long you want balance to last
            print(f"\n🎯 Perfect balance achieved for {steps_in_balance} steps. Episode ending!")
            terminated = True  # Force termination if balance condition is met

        # Print live status at every 100th step
        if step % 100 == 0:
            print(f"Step {step}: Reward = {reward:.3f}, Action = {action}, Pole angle = {pole_angle:.3f}")

        # Control the simulation step rate (60 FPS)
        time.sleep(1.0 / 60.0)

        # Increment the step counter
        step += 1

        # Check if episode is over (either terminated or truncated)
        if terminated or truncated:
            print(f"\n🎯 Episode done. Total reward: {episode_reward:.3f} in {step} steps\n")
            # Reset environment for the next episode
            obs, _ = env.reset()
            step = 0
            episode_reward = 0
            steps_in_balance = 0

finally:
    env.close()  # Ensure the environment is closed properly



# import time
# import gym
# import torch
# import numpy as np
# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# from pendulum_env import InvertedPendulumEnv  # import your custom env

# # Load the trained model
# model = DDPG.load("ddpg_assistive_walker")

# # Create the environment
# env = InvertedPendulumEnv(render=True)
# obs, _ = env.reset()

# # Print action space limits (to ensure actions are within bounds)
# print("Action space low:", env.action_space.low)
# print("Action space high:", env.action_space.high)

# # Add action noise (during inference only for slight movement if stuck)
# # Reduce noise for inference
# action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.05 * np.ones(env.action_space.shape))

# # Variables to track episode and rewards
# step = 0
# episode_reward = 0

# try:
#     while True:
#         # Get action from model
#         action, _ = model.predict(obs, deterministic=True)

#         # Add small noise to avoid freezing (optional)
#         noisy_action = np.clip(action + action_noise(), env.action_space.low, env.action_space.high)

#         # Step through the environment
#         obs, reward, terminated, truncated, info = env.step(noisy_action)
#         episode_reward += reward

#         # Extract angle for display (assume pole angle is at index 2 in observation)
#         pole_angle = obs[2]  # Assuming obs[2] is the pole angle
#         velocity = obs[1]    # Assuming obs[1] is the cart velocity

#         # Optionally, penalize for large pole angle deviations
#         if abs(pole_angle) > 0.1:  # If pole angle is more than 0.1 radians
#             reward -= 0.5  # Apply penalty to the reward

#         # Print live status at every 100th step
#         if step % 100 == 0:
#             print(f"Step {step}: Reward = {reward:.3f}, Action = {action}, Pole angle = {pole_angle:.3f}")

#         # Control the simulation step rate (60 FPS)
#         time.sleep(1.0 / 60.0)

#         # Increment the step counter
#         step += 1

#         # Check if episode is over (either terminated or truncated)
#         if terminated or truncated:
#             print(f"\n🎯 Episode done. Total reward: {episode_reward:.3f} in {step} steps\n")
#             # Reset environment for the next episode
#             obs, _ = env.reset()
#             step = 0
#             episode_reward = 0

# finally:
#     env.close()  # Ensure the environment is closed properly



# import time
# import gym
# import torch
# import numpy as np
# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# from pendulum_env import InvertedPendulumEnv  # import your custom env

# # Load the trained model
# model = DDPG.load("ddpg_assistive_walker")

# # Create the environment
# env = InvertedPendulumEnv(render=True)
# obs, _ = env.reset()

# # Add action noise (during inference only for slight movement if stuck)
# action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape))

# # Variables to track episode and rewards
# step = 0
# episode_reward = 0

# try:
#     while True:
#         # Get action from model
#         action, _ = model.predict(obs, deterministic=True)

#         # Add small noise to avoid freezing (optional)
#         noisy_action = np.clip(action + action_noise(), env.action_space.low, env.action_space.high)

#         # Step through the environment
#         obs, reward, terminated, truncated, info = env.step(noisy_action)
#         episode_reward += reward

#         # Extract angle for display (assume pole angle is at index 2 in observation)
#         pole_angle = obs[2]  # Assuming obs[2] is the pole angle
#         velocity = obs[1]    # Assuming obs[1] is the cart velocity

#         # Print live status at every 100th step
#         if step % 100 == 0:
#             print(f"Step {step}: Reward = {reward:.3f}, Action = {action}, Pole angle = {pole_angle:.3f}")

#         # Control the simulation step rate (60 FPS)
#         time.sleep(1.0 / 60.0)

#         # Increment the step counter
#         step += 1

#         # Check if episode is over (either terminated or truncated)
#         if terminated or truncated:
#             print(f"\n🎯 Episode done. Total reward: {episode_reward:.3f} in {step} steps\n")
#             # Reset environment for the next episode
#             obs, _ = env.reset()
#             step = 0
#             episode_reward = 0

# finally:
#     env.close()  # Ensure the environment is closed properly



# import time
# import gymnasium as gym
# from stable_baselines3 import DDPG

# from pendulum_env import InvertedPendulumEnv  # import your custom env

# # Load environment with GUI rendering
# env = InvertedPendulumEnv(render=True)

# # Load trained model
# model = DDPG.load("ddpg_assistive_walker")

# # Reset the environment
# obs, _ = env.reset()
# total_reward = 0
# step_count = 0

# try:
#     while True:
#         action, _ = model.predict(obs, deterministic=True)
#         obs, reward, terminated, truncated, info = env.step(action)

#         total_reward += reward
#         step_count += 1

#         # Print live status
#         print(f"Step {step_count}: Reward = {reward:.3f}, Action = {action}, Pole angle = {obs[0]:.3f}")

#         time.sleep(1.0 / 60.0)  # 60 FPS display speed

#         if terminated or truncated:
#             print("\nEpisode done.")
#             print(f"Total reward: {total_reward:.2f} | Steps: {step_count}")
#             break
# finally:
#     env.close()

# import gym
# import torch
# import numpy as np
# from stable_baselines3 import DDPG
# from stable_baselines3.common.noise import NormalActionNoise
# import pendulum_env  # Your custom environment module

# # Load the trained model
# model = DDPG.load("ddpg_assistive_walker")

# # Create the environment
# env = InvertedPendulumEnv(render=True)
# obs = env.reset()

# # Add action noise (during inference only for slight movement if stuck)
# action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape), sigma=0.1 * np.ones(env.action_space.shape))

# step = 0
# episode_reward = 0
# while True:
#     # Get action from model
#     action, _ = model.predict(obs, deterministic=True)
    
#     # Add small noise to avoid freezing (optional)
#     noisy_action = np.clip(action + action_noise(), env.action_space.low, env.action_space.high)

#     # Step through environment
#     obs, reward, done, info = env.step(noisy_action)
#     episode_reward += reward

#     # Extract angle for display
#     pole_angle = obs[2]  # Assuming obs[2] is the pole angle
#     velocity = obs[1]    # Assuming obs[1] is cart velocity (or replace accordingly)

#     if step % 100 == 0:
#         print(f"Step {step}: Reward = {reward:.3f}, Action = {action}, Pole angle = {pole_angle:.3f}")


#     step += 1

#     # Reset if episode ends
#     if done:
#         print(f"\n🎯 Episode done. Total reward: {episode_reward:.3f} in {step} steps\n")
#         obs = env.reset()
#         step = 0
#         episode_reward = 0