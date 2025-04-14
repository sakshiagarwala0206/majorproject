import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

class InvertedPendulumEnvGymnasium(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 60}

    def __init__(self, render_mode=None):
        super().__init__()
        self.render_mode = render_mode
        self.physics_client = p.connect(p.GUI if render_mode == "human" else p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)
        p.setGravity(0, 0, -9.8)

        self.urdf_path = "urdf /cartpole.urdf"
        self.cart_id = None

        # Apply torque to prismatic joint (slider)
        self.action_space = spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)

        # Observations: cart_pos, cart_vel, pole_angle, pole_ang_vel
        high = np.array([15.0, 10.0, np.pi, 10.0], dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.8)
        p.setTimeStep(self.time_step)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.1]
        self.cart_id = p.loadURDF(self.urdf_path, start_pos)

        # Disable motors
        p.setJointMotorControl2(self.cart_id, 0, controlMode=p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cart_id, 1, controlMode=p.VELOCITY_CONTROL, force=0)

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def _get_obs(self):
        cart_pos = p.getJointState(self.cart_id, 0)[0]
        cart_vel = p.getJointState(self.cart_id, 0)[1]
        pole_angle = p.getJointState(self.cart_id, 1)[0]
        pole_ang_vel = p.getJointState(self.cart_id, 1)[1]

        return np.array([cart_pos, cart_vel, pole_angle, pole_ang_vel], dtype=np.float32)

    def step(self, action):
        torque = float(np.clip(action[0], -1.0, 1.0)) * 100

        p.setJointMotorControl2(
            bodyUniqueId=self.cart_id,
            jointIndex=0,
            controlMode=p.TORQUE_CONTROL,
            force=torque,
        )

        p.stepSimulation()

        obs = self._get_obs()
        reward = 1.0 - (abs(obs[2]) / (np.pi / 2))  # normalized upright reward
        terminated = abs(obs[2]) > np.pi / 2 or abs(obs[0]) > 15.0
        truncated = False

        return obs, reward, terminated, truncated, {}

    def render(self):
        if self.render_mode == "human":
            p.resetDebugVisualizerCamera(
                cameraDistance=3, cameraYaw=0, cameraPitch=-30, cameraTargetPosition=[0, 0, 0.5]
            )

    def close(self):
        p.disconnect()
