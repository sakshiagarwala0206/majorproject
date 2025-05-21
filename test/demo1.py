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

from environments.walker_1 import AssistiveWalkerContinuousEnv

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

# Add a debug slider for disturbance torque (safe range)
slider_id = p.addUserDebugParameter("Disturbance Torque (Nm)", -20, 20, 0)
button_left = p.addUserDebugParameter("Nudge Left", 1, 0, 0)
button_right = p.addUserDebugParameter("Nudge Right", 1, 0, 0)

obs_labels = [
    "pole_pitch_angle", "pole_pitch_velocity", "base_x", "base_y", "base_yaw",
    "left_wheel_vel", "right_wheel_vel",
    "imu_roll", "imu_pitch", "imu_yaw",
    "imu_ang_vel_x", "imu_ang_vel_y", "imu_ang_vel_z",
    "imu_lin_acc_x", "imu_lin_acc_y", "imu_lin_acc_z"
]

print("\n=== SAFE DEMO MODE ===")
print("Use the 'Disturbance Torque' slider for gentle disturbance.")
print("Or press 'Nudge Left/Right' buttons for a small, bounded push.")
print("======================\n")

done = False
step = 0
while not done:
    # Read safe disturbance value from slider
    torque = p.readUserDebugParameter(slider_id)
    robot_id = env.unwrapped.robot_id  # Use unwrapped to access custom attribute
    if abs(torque) > 1e-3:
        p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=torque)
        print(f"Applied safe disturbance: {torque:.2f} Nm")
        # Reset slider to 0 after applying
        # p.addUserDebugParameter("Disturbance Torque (Nm)", -3, 3, 0)

    # Check for button presses
    if p.readUserDebugParameter(button_left) > 0.5:
        p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=-2.0)
        print("Nudge Left")
       
    if p.readUserDebugParameter(button_right) > 0.5:
        p.setJointMotorControl2(robot_id, 2, p.TORQUE_CONTROL, force=2.0)
        print("Nudge RiGHT")
       

    # RL model acts
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
        position, orientation = p.getBasePositionAndOrientation(robot_id)
        print(f"Position: {position}, Orientation: {orientation}")
    except AttributeError:
        pass

    time.sleep(1.0 / 60.0)
    step += 1

env.close()
