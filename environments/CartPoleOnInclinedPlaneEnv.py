import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class InclinedCartPoleBaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    gui_connected = False

    def __init__(self, render_mode=False, incline_angle=0.1):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.render_fps = 60
        self.incline_angle = incline_angle  # radians
        self.cartpole_id = None

        if render_mode:
            if not InclinedCartPoleBaseEnv.gui_connected:
                p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                InclinedCartPoleBaseEnv.gui_connected = True
            else:
                p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)

        # Load an inclined plane
        p.loadURDF("plane.urdf", baseOrientation=p.getQuaternionFromEuler([0, self.incline_angle, 0]))

        # Load the cartpole on inclined surface
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

    def _get_obs(self):
        cart_state = p.getJointState(self.cartpole_id, 0)
        cart_position, cart_velocity = cart_state[0], cart_state[1]
        pole_state = p.getJointState(self.cartpole_id, 1)
        pole_angle, pole_velocity = pole_state[0], pole_state[1]
        # self.logger.record('rollout/pole_angle', pole_angle)
        # self.logger.record('rollout/reward', self._get_reward([pole_angle, pole_velocity, cart_position, cart_velocity]))
        return np.array([pole_angle, pole_velocity, cart_position, cart_velocity], dtype=np.float32)

    def _get_reward(self, obs):
        cart_pos, cart_vel, pole_angle, pole_vel = obs
        reward = 0.0
        # Normalize for gentle scaling
        pole_angle_penalty = (pole_angle / self.max_pole_angle) ** 2
        pole_velocity_penalty = (pole_vel / self.max_pole_vel) ** 2
        cart_pos_penalty = (cart_pos / self.max_cart_pos) ** 2

        # Weighted combination
        reward = 1.0 \
             - 0.5 * pole_angle_penalty \
             - 0.3 * pole_velocity_penalty \
             - 0.2 * cart_pos_penalty

    # # Optional: clip or zero out if episode ends
    # if abs(pole_angle) > self.max_pole_angle:
    #     reward = -10.0  # heavy penalty for falling
    
    # # Bonus for keeping the pole near upright
    #     if abs(pole_angle) < 0.1:
    #         reward += 1  # Small positive reward for staying upright
    
        return reward
    

    def _is_done(self, obs):
        pole_angle = obs[0]
        cart_position = obs[2]
        return abs(pole_angle) > 0.5 or abs(cart_position) > 2.4

    def close(self):
        p.disconnect()

class InclinedCartPoleDiscreteEnv(InclinedCartPoleBaseEnv):
    def __init__(self, render_mode=False, incline_angle=0.1):
        super().__init__(render_mode, incline_angle)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.max_pole_angle = np.deg2rad(15)  # or another value that makes sense

    def step(self, action):
        torque = 10.0 if action == 1 else -10.0
        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}

class InclinedCartPoleContinuousEnv(InclinedCartPoleBaseEnv):
    def __init__(self, render_mode=False, incline_angle=0.1):
        super().__init__(render_mode, incline_angle)
        self.action_space = spaces.Box(low=np.array([-5.0]), high=np.array([5.0]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )
        self.max_pole_angle = np.deg2rad(15)  # or another value that makes sense
        self.max_pole_vel = 10.0  
        self.max_cart_pos = 2.4

    def step(self, action):
        cart_torque = float(action[0])
        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=cart_torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}
