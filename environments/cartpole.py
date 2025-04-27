import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class CartPoleBaseEnv(gymnasium.Env):
    def __init__(self, render=False):
        super(CartPoleBaseEnv, self).__init__()
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.cartpole_id = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        urdf_path = os.path.join(project_root, 'urdf', 'cartpole.urdf')

        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    
        self.cartpole_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])

        p.setJointMotorControl2(self.cartpole_id, 0, p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole_id, 1, p.VELOCITY_CONTROL, force=0)

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def step(self, action):
        raise NotImplementedError("step() must be implemented in subclasses")

    def _get_obs(self):
        cart_state = p.getJointState(self.cartpole_id, 0)
        cart_position = cart_state[0]
        cart_velocity = cart_state[1]

        pole_state = p.getJointState(self.cartpole_id, 1)
        pole_angle = pole_state[0]
        pole_velocity = pole_state[1]

        return np.array([pole_angle, pole_velocity, cart_position, cart_velocity], dtype=np.float32)

    def _get_reward(self, obs):
        pole_angle = obs[0]
        return 1.0 - (abs(pole_angle) / np.pi)

    def _is_done(self, obs):
        pole_angle = obs[0]
        cart_position = obs[2]
        return abs(pole_angle) > 0.5 or abs(cart_position) > 2.4

    def close(self):
        p.disconnect()

class CartPoleDiscreteEnv(CartPoleBaseEnv):
    def __init__(self, render=False):
        super(CartPoleDiscreteEnv, self).__init__(render)
        self.action_space = spaces.Discrete(2)  # Actions: 0 = left, 1 = right

    def step(self, action):
        torque = 10.0 if action == 1 else -10.0  # Apply torque based on discrete action (left or right)

        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}

class CartPoleContinuousEnv(CartPoleBaseEnv):
    def __init__(self, render=False):
        super(CartPoleContinuousEnv, self).__init__(render)
        self.action_space = spaces.Box(low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]), dtype=np.float32)

    def step(self, action):
        cart_torque = float(action[0])
        pole_torque = float(action[1])

        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=cart_torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=pole_torque)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}
