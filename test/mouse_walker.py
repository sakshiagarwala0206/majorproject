import wandb

wandb.init(
    project="assistive-walker-mouse-test",
    name="mouse_disturbance_run"
)

import gymnasium
import pybullet as p
import pybullet_data
import numpy as np
import time
import os
import sys
from stable_baselines3 import PPO
from gymnasium.envs.registration import register

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environments.walker_1 import AssistiveWalkerContinuousEnv

register(
    id="AssistiveWalkerContinuousEnv-v0",
    entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
    max_episode_steps=100000,
)

env = gymnasium.make("AssistiveWalkerContinuousEnv-v0", render_mode="human")
obs, _ = env.reset()
print("Observation shape at reset:", obs.shape)

model = PPO.load("models/ppo/ppo_20250517_020412_assistivewalker_final.zip")

if p.getConnectionInfo()['isConnected'] == 0:
    p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

print("\n=== INSTRUCTIONS ===")
print("Drag the walker/pole/handle with your mouse in the PyBullet GUI to apply a disturbance.")
print("The RL model will observe the state and apply corrective action at every step.")
print("====================\n")

obs_labels = [
    "pole_pitch_angle", "pole_pitch_velocity", "base_x", "base_y", "base_yaw",
    "left_wheel_vel", "right_wheel_vel",
    "imu_roll", "imu_pitch", "imu_yaw",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"
]

done = False
step = 0
while not done:
    if obs.shape != model.observation_space.shape:
        raise ValueError(
            f"Model expects obs of shape {model.observation_space.shape}, but env returned {obs.shape}."
            " Make sure your environment and model are both using 16D observations."
        )

    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # Print all observation values with labels
    print(f"\nStep {step}")
    print("Action (wheel torques):", action)
    for i, label in enumerate(obs_labels):
        print(f"{label}: {obs[i]:.4f}")
    print(f"Reward: {reward:.4f}")

    try:
        robot_id = env.robot_id
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        print(f"Position: {position}, Orientation: {orientation}")
    except AttributeError:
        pass

    time.sleep(1.0 / 60.0)
    step += 1

env.close()



# import wandb

# wandb.init(
#     project="assistive-walker-mouse-test",
#     name="mouse_disturbance_run"
# )

# import gymnasium
# import pybullet as p
# import pybullet_data
# import numpy as np
# import time
# import os
# import sys
# from stable_baselines3 import PPO
# from gymnasium.envs.registration import register

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from environments.walker_1 import AssistiveWalkerContinuousEnv

# register(
#     id="AssistiveWalkerContinuousEnv-v0",
#     entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
#     max_episode_steps=100000,
# )

# env = gymnasium.make("AssistiveWalkerContinuousEnv-v0", render_mode="human", test_mode=True)
# obs, _ = env.reset()
# print("Observation shape at reset:", obs.shape)

# model = PPO.load("models/ppo/BaseConfig_PPO_AW_01.zip")

# if p.getConnectionInfo()['isConnected'] == 0:
#     p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# print("\n=== INSTRUCTIONS ===")
# print("Drag the walker/pole/handle with your mouse in the PyBullet GUI to apply a disturbance.")
# print("The RL model will observe the state and apply corrective action at every step.")
# print("====================\n")

# done = False
# step = 0
# while not done:
#     if obs.shape[0] != model.observation_space.shape[0]:
#         raise ValueError(
#             f"Model expects obs of shape {model.observation_space.shape}, but env returned {obs.shape}."
#             " Make sure your environment and model are both using 16D observations."
#         )

#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     try:
#         robot_id = env.robot_id
#         position, orientation = p.getBasePositionAndOrientation(robot_id)
#         print(f"Step {step} | Position: {position}, Orientation: {orientation}")
#     except AttributeError:
#         pass

#     time.sleep(1.0 / 60.0)
#     step += 1

# env.close()
