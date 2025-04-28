import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

# ------------------ Discrete Environment ------------------ #
class CartPoleVelocityControlEnvDiscrete(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=False, test_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.test_mode = test_mode

        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1.0 / 240.0)

        self.cartpole_id = None
        self.action_space = spaces.Discrete(2)  # 0 = move left, 1 = move right
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

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

        p.setJointMotorControl2(self.cartpole_id, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.cartpole_id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def step(self, action):
        target_velocity = 5.0 if action == 1 else -5.0

        p.setJointMotorControl2(self.cartpole_id, 0, p.VELOCITY_CONTROL, targetVelocity=target_velocity, force=10.0)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        cart_state = p.getJointState(self.cartpole_id, 0)
        pole_state = p.getJointState(self.cartpole_id, 1)

        return np.array([
            pole_state[0],
            pole_state[1],
            cart_state[0],
            cart_state[1]
        ], dtype=np.float32)

    def _get_reward(self, obs):
        pole_angle = obs[0]
        cart_velocity = obs[3]
        return -abs(pole_angle) - 0.1 * abs(cart_velocity)

    def _is_done(self, obs):
        pole_angle = obs[0]
        cart_position = obs[2]
        return abs(pole_angle) > 0.5 or abs(cart_position) > 2.4

    def close(self):
        p.disconnect()

# ------------------ Continuous Environment ------------------ #
class CartPoleVelocityControlEnvContinuous(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}

    def __init__(self, render_mode=False, test_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.test_mode = test_mode

        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setTimeStep(1.0 / 240.0)

        self.cartpole_id = None
        self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

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

        p.setJointMotorControl2(self.cartpole_id, 0, p.VELOCITY_CONTROL, targetVelocity=0, force=0)
        p.setJointMotorControl2(self.cartpole_id, 1, p.VELOCITY_CONTROL, targetVelocity=0, force=0)

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def step(self, action):
        target_velocity = float(action[0])

        p.setJointMotorControl2(self.cartpole_id, 0, p.VELOCITY_CONTROL, targetVelocity=target_velocity, force=10.0)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        cart_state = p.getJointState(self.cartpole_id, 0)
        pole_state = p.getJointState(self.cartpole_id, 1)

        return np.array([
            pole_state[0],
            pole_state[1],
            cart_state[0],
            cart_state[1]
        ], dtype=np.float32)

    def _get_reward(self, obs):
        pole_angle = obs[0]
        cart_velocity = obs[3]
        return -abs(pole_angle) - 0.1 * abs(cart_velocity)

    def _is_done(self, obs):
        pole_angle = obs[0]
        cart_position = obs[2]
        return abs(pole_angle) > 0.5 or abs(cart_position) > 2.4

    def close(self):
        p.disconnect()
