import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data

class AssistiveWalkerEnv1(gymnasium.Env):
    def __init__(self, render=False):
        super(AssistiveWalkerEnv1, self).__init__()
        self.render_mode = render
        if self.render_mode:
            p.connect(p.GUI)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.walker_id = None

        # Apply torques to both wheels
        self.action_space = spaces.Box(low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]), dtype=np.float32)

        # Observation: [pole_angle, pole_velocity, cart_velocity]
        obs_high = np.array([np.pi, np.inf, np.inf], dtype=np.float32)
        self.observation_space = spaces.Box(-obs_high, obs_high, dtype=np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        start_pos = [0, 0, 0.1]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.walker_id = p.loadURDF("chasis1.urdf", start_pos, start_ori)

        # Disable default motor control
        p.setJointMotorControl2(self.walker_id, 0, p.VELOCITY_CONTROL, force=0)  # left wheel
        p.setJointMotorControl2(self.walker_id, 1, p.VELOCITY_CONTROL, force=0)  # right wheel

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs(), {}

    def step(self, action):
        left_torque = float(action[0])
        right_torque = float(action[1])

        # Apply torque to wheels
        p.setJointMotorControl2(self.walker_id, 0, p.TORQUE_CONTROL, force=left_torque)
        p.setJointMotorControl2(self.walker_id, 1, p.TORQUE_CONTROL, force=right_torque)

        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False

        return obs, reward, terminated, truncated, {}

    def _get_obs(self):
        # Joint 2 is the pole joint
        pole_state = p.getJointState(self.walker_id, 2)
        pole_angle = pole_state[0]        # radians
        pole_velocity = pole_state[1]     # angular velocity

        linear_vel, _ = p.getBaseVelocity(self.walker_id)
        base_x_velocity = linear_vel[0]

        return np.array([pole_angle, pole_velocity, base_x_velocity], dtype=np.float32)

    def _get_reward(self, obs):
        pole_angle = obs[0]
        return 1.0 - (abs(pole_angle) / np.pi)  # closer to upright, higher the reward

    def _is_done(self, obs):
        pole_angle = obs[0]
        return abs(pole_angle) > 0.5  # failure if it tilts too much

    def close(self):
        p.disconnect()
