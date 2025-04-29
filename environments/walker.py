import gym
import numpy as np
import pybullet as p
import pybullet_data
from gym import spaces

class CartPoleWheelsEnv(gym.Env):
    def __init__(self, render=False):
        super(CartPoleWheelsEnv, self).__init__()
        
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)
        
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.81)

        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        # Load plane and robot
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("path_to_your_urdf/cart_pole_with_wheels.urdf", [0, 0, 0.1])

        # Joint indices
        self.left_wheel_joint = 0
        self.right_wheel_joint = 1
        self.pole_joint = 2

        # Action space: torques to apply on wheels
        torque_limit = 5.0
        self.action_space = spaces.Box(low=-torque_limit, high=torque_limit, shape=(2,), dtype=np.float32)

        # Observation space: [pole angle, pole angular velocity, cart linear velocity]
        high = np.array([np.pi, np.finfo(np.float32).max, np.finfo(np.float32).max, np.finfo(np.float32).max])
        self.observation_space = spaces.Box(low=-high, high=high, dtype=np.float32)

        self.max_steps = 500
        self.step_counter = 0

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        self.plane_id = p.loadURDF("plane.urdf")
        self.robot_id = p.loadURDF("path_to_your_urdf/cart_pole_with_wheels.urdf", [0, 0, 0.1])

        # Reset joints
        p.resetJointState(self.robot_id, self.left_wheel_joint, 0, 0)
        p.resetJointState(self.robot_id, self.right_wheel_joint, 0, 0)
        p.resetJointState(self.robot_id, self.pole_joint, 0, 0)

        self.step_counter = 0
        return self._get_obs()

    def step(self, action):
        left_torque, right_torque = action

        # Apply torque to left and right wheels
        p.setJointMotorControl2(self.robot_id, self.left_wheel_joint, controlMode=p.TORQUE_CONTROL, force=left_torque)
        p.setJointMotorControl2(self.robot_id, self.right_wheel_joint, controlMode=p.TORQUE_CONTROL, force=right_torque)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._compute_reward(obs)
        done = self._check_done(obs)

        self.step_counter += 1
        if self.step_counter >= self.max_steps:
            done = True

        return obs, reward, done, {}

    def _get_obs(self):
        # Pole state
        pole_info = p.getJointState(self.robot_id, self.pole_joint)
        pole_angle = pole_info[0]
        pole_angular_velocity = pole_info[1]

        # Base (cart) linear velocity
        base_velocity = p.getBaseVelocity(self.robot_id)[0][0]  # x-axis velocity

        return np.array([pole_angle, pole_angular_velocity, base_velocity], dtype=np.float32)

    def _compute_reward(self, obs):
        pole_angle = obs[0]
        base_velocity = obs[2]

        # Reward for keeping pole vertical and cart slow
        reward = 1.0 - (abs(pole_angle) + 0.1 * abs(base_velocity))
        return reward

    def _check_done(self, obs):
        pole_angle = obs[0]
        if abs(pole_angle) > 0.5:  # ~30 degrees
            return True
        return False

    def render(self, mode='human'):
        pass  # Already handled in __init__

    def close(self):
        p.disconnect()
