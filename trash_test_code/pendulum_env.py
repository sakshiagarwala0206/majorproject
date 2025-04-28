import gym
from gym import spaces
import numpy as np
import pybullet as p
import pybullet_data
import time

class InvertedPendulumEnv(gym.Env):
    def __init__(self, render=False):
        super(InvertedPendulumEnv, self).__init__()

        self.render_mode = render
        if self.render_mode:
            p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.cartpole_id = None

        # Action space: a continuous scalar to control cart velocity
        self.action_space = spaces.Box(low=np.array([-10.0]), high=np.array([10.0]), dtype=np.float32)

        # Observation space: [cart_pos, cart_vel, pole_angle, pole_vel]
        high = np.array([np.inf]*4, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

    def reset(self):
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")
        start_pos = [0, 0, 0.1]
        start_ori = p.getQuaternionFromEuler([0, 0, 0])
        self.cartpole_id = p.loadURDF("pendulum.urdf", start_pos, start_ori, useFixedBase=False)

        # Disable default motor control for both joints
        p.setJointMotorControl2(self.cartpole_id, 0, controlMode=p.VELOCITY_CONTROL, force=0)
        p.setJointMotorControl2(self.cartpole_id, 1, controlMode=p.VELOCITY_CONTROL, force=0)

        for _ in range(10):
            p.stepSimulation()

        return self._get_obs()

    def step(self, action):
        force = float(action[0])

        # Apply velocity control to cart (joint 0, prismatic)
        p.setJointMotorControl2(
        bodyUniqueId=self.cartpole_id,
        jointIndex=0,
        controlMode=p.TORQUE_CONTROL,
        force=force
    ) 


        p.stepSimulation()

        obs = self._get_obs()
        reward = self._get_reward(obs)
        done = self._is_done(obs)

        return obs, reward, done, {}

    def _get_obs(self):
        # Joint 0 = cart (slider), Joint 1 = pole (hinge)
        cart_info = p.getJointState(self.cartpole_id, 0)
        pole_info = p.getJointState(self.cartpole_id, 1)

        cart_pos = cart_info[0]
        cart_vel = cart_info[1]
        pole_angle = pole_info[0]
        pole_vel = pole_info[1]

        return np.array([cart_pos, cart_vel, pole_angle, pole_vel], dtype=np.float32)

    def _get_reward(self, obs):
        # Reward is highest when pole is upright (angle near 0)
        pole_angle = obs[2]
        return 1.0 - (abs(pole_angle) / np.pi)

    def _is_done(self, obs):
        cart_pos, _, pole_angle, _ = obs
        return abs(cart_pos) > 2.4 or abs(pole_angle) > 0.2

    def close(self):
        p.disconnect()
