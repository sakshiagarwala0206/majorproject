import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class InclinedCartPoleBaseEnv(gymnasium.Env):
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
        plane_id = p.loadURDF("plane.urdf", baseOrientation=p.getQuaternionFromEuler([0, self.incline_angle, 0]))

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
        return np.array([pole_angle, pole_velocity, cart_position, cart_velocity], dtype=np.float32)

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

class InclinedCartPoleDiscreteEnv(InclinedCartPoleBaseEnv):
    def __init__(self, render_mode=False, incline_angle=0.1):
        super().__init__(render_mode, incline_angle)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

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
