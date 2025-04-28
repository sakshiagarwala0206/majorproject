import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class CartPoleBaseEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    gui_connected = False

    def __init__(self, render_mode=None, test_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.render_fps = 60
        self.test_mode = test_mode

        if self.render_mode:
            if not CartPoleBaseEnv.gui_connected:
                p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                CartPoleBaseEnv.gui_connected = True
            else:
                p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.cartpole_id = None
        self.current_step = 0

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

        return self._get_obs(), {}

    def _get_obs(self):
        cart_state = p.getJointState(self.cartpole_id, 0)
        cart_position = cart_state[0]
        cart_velocity = cart_state[1]

        pole_state = p.getJointState(self.cartpole_id, 1)
        pole_angle = pole_state[0]
        pole_velocity = pole_state[1]

        # Example realistic sensor data
        sensor_data = self._get_sensors_data(cart_velocity, pole_velocity, pole_angle)

        return np.concatenate([np.array([pole_angle, pole_velocity, cart_position, cart_velocity]), sensor_data], axis=0)

    def _get_reward(self, obs):
        pole_angle = obs[0]
        cart_velocity = obs[3]
        return -abs(pole_angle) - 0.1 * abs(cart_velocity)

    def _is_done(self, obs):
        pole_angle = obs[0]
        cart_position = obs[2]
        return abs(pole_angle) > 0.5 or abs(cart_position) > 2.4

    def _get_sensors_data(self, cart_velocity, pole_velocity, pole_angle):
        # Accelerometer simulation (X, Y, Z)
        # For simplicity, assume acceleration is related to the cart and pole velocities
        accelerometer_x = cart_velocity * 0.1 + np.random.normal(0, 0.02)
        accelerometer_y = pole_velocity * 0.05 + np.random.normal(0, 0.02)
        accelerometer_z = np.sin(pole_angle) * 9.81 + np.random.normal(0, 0.02)  # Gravity-based

        # Gyroscope simulation (angular velocity of the pole)
        gyroscope = pole_velocity + np.random.normal(0, 0.05)  # Adding noise to simulate imperfect measurement

        # Force sensor simulation: Simulate forces on the cart due to motor torque
        force_sensor_data = np.random.uniform(-1.0, 1.0, 5)  # Random noise for force sensors

        # Proximity sensors simulation: Measure distances to an imaginary obstacle
        proximity_sensor_data = np.random.uniform(0, 1, 3)  # Random values between 0 and 1 to represent distance

        return np.concatenate([accelerometer_x, accelerometer_y, accelerometer_z, gyroscope, force_sensor_data, proximity_sensor_data])

    def close(self):
        p.disconnect()

class CartPoleWithSensorsDiscreteEnv(CartPoleBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -1.0, -1.0, -9.81, -np.inf, -1.0, 0.0], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf, 1.0, 1.0, 9.81, np.inf, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        torque = 10.0 if action == 1 else -10.0

        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=0.0)

        p.stepSimulation()

        self.current_step += 1

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}

class CartPoleWithSensorsContinuousEnv(CartPoleBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.action_space = spaces.Box(low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -1.0, -1.0, -9.81, -np.inf, -1.0, 0.0], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf, 1.0, 1.0, 9.81, np.inf, 1.0, 1.0], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        cart_torque = float(action[0])
        pole_torque = float(action[1])

        p.setJointMotorControl2(self.cartpole_id, 0, p.TORQUE_CONTROL, force=cart_torque)
        p.setJointMotorControl2(self.cartpole_id, 1, p.TORQUE_CONTROL, force=pole_torque)

        p.stepSimulation()

        self.current_step += 1

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}
