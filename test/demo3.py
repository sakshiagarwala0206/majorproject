import wandb
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
    entry_point="environments.walker:AssistiveWalkerContinuousEnv",
    max_episode_steps=10000000,
)

# Initialize WandB first
wandb.init(
    project="assistive-walker-mouse-test",
    name="mouse_disturbance_run"
)

env = gymnasium.make("AssistiveWalkerContinuousEnv-v0", render_mode="human")
obs, _ = env.reset()
model = PPO.load("final_model/ppo/Final_PPO.zip")

if p.getConnectionInfo()['isConnected'] == 0:
    p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# GUI Elements
slider_id = p.addUserDebugParameter("Disturbance Torque (Nm)", -20, 20, 0)
angle_slider = p.addUserDebugParameter("Pole Angle (rad)", -1.0, 1.0, 0.0)

# State variables
disturbance_active = False
disturbance_torque = 0.0
DISTURBANCE_DURATION = 1  # ~0.2s at 240Hz
current_disturbance_steps = 0

last_angle = None
print_debug = False  # Set to True for verbose logging

robot_id = env.unwrapped.robot_id

done = False
step = 0
while not done:
    # 1. Handle angle updates only when slider changes
    desired_angle = p.readUserDebugParameter(angle_slider)
    if last_angle is None or abs(desired_angle - last_angle) > 0.01:
        p.resetJointState(robot_id, 2, desired_angle, 0)
        if print_debug:
            print(f"Set pole angle to {desired_angle:.3f} rad")
        last_angle = desired_angle

    # 2. Handle disturbances with cooldown
    torque = p.readUserDebugParameter(slider_id)
    if abs(torque) > 1e-3 and not disturbance_active:
        disturbance_torque = torque
        disturbance_active = True
        current_disturbance_steps = 0
        if print_debug:
            print(f"Applying disturbance: {torque:.2f} Nm for {DISTURBANCE_DURATION} steps")

    if disturbance_active:
        p.applyExternalTorque(robot_id, 2, [0, disturbance_torque, 0], p.LINK_FRAME)

        current_disturbance_steps += 1
        if current_disturbance_steps >= DISTURBANCE_DURATION:
            disturbance_active = False

    # 3. RL model acts
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated

    # 4. Controlled logging
    if step % 10 == 0 and print_debug:  # Log every 10 steps
        print(f"\nStep {step}")
        print(f"Action (wheel torques): {action}")
        print(f"IMU Pitch: {obs[8]:.4f}, Roll: {obs[7]:.4f}")
        print(f"Reward: {reward:.4f}")

    time.sleep(1.0 / 60.0)
    step += 1

env.close()
