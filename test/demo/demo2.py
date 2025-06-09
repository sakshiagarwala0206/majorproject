import wandb

wandb.init(
    project="assistive-walker-mouse-test",
    name="mouse_disturbance_run"
)

import pybullet as p
import pybullet_data
import time
from stable_baselines3 import PPO
import gymnasium
import os
import sys
from gymnasium.envs.registration import register

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environments.walker import AssistiveWalkerContinuousEnv

register(
    id="AssistiveWalkerContinuousEnv-v0",
    entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
    max_episode_steps=100000,
)

env = gymnasium.make("AssistiveWalkerContinuousEnv-v0", render_mode="human")
obs, _ = env.reset()
model = PPO.load("models/ppo/BaseConfig_PPO_AW_01.zip")

if p.getConnectionInfo()['isConnected'] == 0:
    p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Add debug sliders
slider_id = p.addUserDebugParameter("Disturbance Torque (Nm)", -1000, 1000, 0)
angle_slider = p.addUserDebugParameter("Pole Angle (rad)", -1.0, 1.0, 0.0)  # -1 to 1 radian

obs_labels = [
    "pole_pitch_angle", "pole_pitch_velocity", "base_x", "base_y", "base_yaw",
    "left_wheel_vel", "right_wheel_vel",
    "imu_roll", "imu_pitch", "imu_yaw",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"
]

print("\n=== SAFE DEMO MODE ===")
print("Use the 'Disturbance Torque' slider for a visible disturbance (hold for effect).")
print("Use the 'Pole Angle (rad)' slider to set the pole angle directly.")
print("======================\n")

disturbance_steps = 0
disturbance_torque = 0.0
DISTURBANCE_DURATION = 1  # ~1s at 240Hz

done = False
step = 0
while not done:
    robot_id = env.unwrapped.robot_id

    # --- Interactive pole angle set ---
    desired_angle = p.readUserDebugParameter(angle_slider)
    # Set the pole angle (joint 2) to the slider value (velocity=0)
    p.resetJointState(robot_id, 2, desired_angle, 0)
    print(f"Set pole angle to {desired_angle:.3f} rad")

    # --- Disturbance torque ---
    torque = p.readUserDebugParameter(slider_id)
    if abs(torque) > 1e-3 and disturbance_steps == 0:
        disturbance_torque = torque
        disturbance_steps = DISTURBANCE_DURATION
        print(f"Applying disturbance: {torque:.2f} Nm for {DISTURBANCE_DURATION} steps")

    if disturbance_steps > 0:
        p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=disturbance_torque)
        disturbance_steps -= 1
    else:
        p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=0)

    # RL model acts on wheels
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    print(f"\nStep {step}")
    print("Action (wheel torques):", action)
    for i, label in enumerate(obs_labels):
        print(f"{label}: {obs[i]:.4f}")
    print(f"Reward: {reward:.4f}")
    print(f"Disturbance torque on pole: {disturbance_torque if disturbance_steps > 0 else 0:.2f} Nm")

    try:
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

# import pybullet as p
# import pybullet_data
# import time
# from stable_baselines3 import PPO
# import gymnasium
# import os
# import sys
# from gymnasium.envs.registration import register

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# from environments.walker_1 import AssistiveWalkerContinuousEnv

# register(
#     id="AssistiveWalkerContinuousEnv-v0",
#     entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
#     max_episode_steps=100000,
# )

# env = gymnasium.make("AssistiveWalkerContinuousEnv-v0", render_mode="human")
# obs, _ = env.reset()
# model = PPO.load("models/ppo/BaseConfig_PPO_AW_01.zip")

# if p.getConnectionInfo()['isConnected'] == 0:
#     p.connect(p.GUI)
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Add a debug slider for disturbance torque (safe range)
# slider_id = p.addUserDebugParameter("Disturbance Torque (Nm)", -1000, 1000, 0)

# obs_labels = [
#     "pole_pitch_angle", "pole_pitch_velocity", "base_x", "base_y", "base_yaw",
#     "left_wheel_vel", "right_wheel_vel",
#     "imu_roll", "imu_pitch", "imu_yaw",
#     "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
#     "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"
# ]

# print("\n=== SAFE DEMO MODE ===")
# print("Use the 'Disturbance Torque' slider for a visible disturbance (hold for effect).")
# print("======================\n")

# # Disturbance state variables
# disturbance_steps = 0
# disturbance_torque = 0.0
# DISTURBANCE_DURATION = 240  # ~0.2 seconds at 240Hz

# done = False
# step = 0
# while not done:
#     robot_id = env.unwrapped.robot_id

#     # Read disturbance from slider
#     torque = p.readUserDebugParameter(slider_id)
#     # If user sets slider (and no disturbance is active), apply for DISTURBANCE_DURATION steps
#     if abs(torque) > 1e-3 and disturbance_steps == 0:
#         disturbance_torque = torque
#         disturbance_steps = DISTURBANCE_DURATION
#         print(f"Applying disturbance: {torque:.2f} Nm for {DISTURBANCE_DURATION} steps")

#     # Apply disturbance if active
#     if disturbance_steps > 0:
#         p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=disturbance_torque)
#         disturbance_steps -= 1
#     else:
#         # No disturbance on pole joint
#         p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=0)

#     # RL model acts on wheels
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)
#     done = terminated or truncated

#     # Print all observation values with labels
#     print(f"\nStep {step}")
#     print("Action (wheel torques):", action)
#     for i, label in enumerate(obs_labels):
#         print(f"{label}: {obs[i]:.4f}")
#     print(f"Reward: {reward:.4f}")
#     print(f"Disturbance torque on pole: {disturbance_torque if disturbance_steps > 0 else 0:.2f} Nm")

#     try:
#         position, orientation = p.getBasePositionAndOrientation(robot_id)
#         print(f"Position: {position}, Orientation: {orientation}")
#     except AttributeError:
#         pass

#     time.sleep(1.0 / 60.0)
#     step += 1

# env.close()
