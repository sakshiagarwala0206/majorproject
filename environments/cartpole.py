import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class CartPoleBaseEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    # Class-level flag for GUI connection
    gui_connected = False

    def __init__(self, render_mode=None, test_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.render_fps = 60
        self.test_mode = test_mode  # Add test mode flag

        if self.render_mode:
            if not CartPoleBaseEnv.gui_connected:  # Connect only once
                p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                CartPoleBaseEnv.gui_connected = True
            else:
                p.connect(p.DIRECT)  # For other environments, use DIRECT mode
        else:
            p.connect(p.DIRECT)  # Non-GUI mode

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.cartpole_id = None
        self.current_step = 0
        self.disturbance_step = None
        self.disturbed = False
        self.steps_after_disturbance = 0
        self.reward_after_disturbance = 0.0

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

        self.current_step = 0
        self.disturbed = False
        self.steps_after_disturbance = 0
        self.reward_after_disturbance = 0.0

        if self.test_mode:
            # Set when disturbance will happen during test (fixed/random)
            self.disturbance_step = np.random.randint(100, 300)

        if self.render_mode:
            p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)  # Allow manual disturbance

        return self._get_obs(), {}

    def disturb_pole(self, method="torque", torque_magnitude=5.0, angle_range=0.1):
        if method == "torque":
            random_torque = np.random.uniform(-torque_magnitude, torque_magnitude)
            p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=random_torque)
        elif method == "tilt":
            random_angle = np.random.uniform(-angle_range, angle_range)
            current_velocity = p.getJointState(self.cartpole_id, 1)[1]
            p.resetJointState(self.cartpole_id, 1, random_angle, current_velocity)
        else:
            raise ValueError("Unknown disturbance method: choose 'torque' or 'tilt'")

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
    metadata = {"render_modes": ["human", "rgb_array"]}
    
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.action_space = spaces.Discrete(2)  # Actions: 0 = left, 1 = right
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
    def step(self, action):
        torque = 10.0 if action == 1 else -10.0  # Apply torque based on discrete action (left or right)

        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        self.current_step += 1

        # Apply disturbance if in test mode and correct step
        if self.test_mode and self.current_step == self.disturbance_step:
            self.disturb_pole(method="torque")

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        # Record metrics after disturbance
        if self.test_mode and self.current_step >= self.disturbance_step:
            self.disturbed = True

        if self.disturbed:
            self.steps_after_disturbance += 1
            self.reward_after_disturbance += reward

        return obs, reward, terminated, truncated, {}

class CartPoleContinuousEnv(CartPoleBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.action_space = spaces.Box(low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        
    def step(self, action):
        cart_torque = float(action[0])
        pole_torque = float(action[1])

        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=cart_torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=pole_torque)

        p.stepSimulation()

        self.current_step += 1

        # Apply disturbance if in test mode and correct step
        if self.test_mode and self.current_step == self.disturbance_step:
            self.disturb_pole(method="torque")

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        # Record metrics after disturbance
        if self.test_mode and self.current_step >= self.disturbance_step:
            self.disturbed = True

        if self.disturbed:
            self.steps_after_disturbance += 1
            self.reward_after_disturbance += reward

        return obs, reward, terminated, truncated, {}
