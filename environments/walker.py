import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

class AssistiveWalkerBaseEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"]}
    gui_connected = False

    def __init__(self, render_mode=None, test_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.viewer = None
        self.render_fps = 60
        self.test_mode = test_mode

        if self.render_mode:
            if not AssistiveWalkerBaseEnv.gui_connected:
                p.connect(p.GUI)
                AssistiveWalkerBaseEnv.gui_connected = True
            else:
                p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)

        self.robot_id = None
        self.current_step = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # Adjust path to your walker URDF
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at {urdf_path}")

        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.15])

        # Disable default motors for wheels
        for joint in [0, 1]:  # Update indices if necessary
            p.setJointMotorControl2(self.robot_id, joint, p.VELOCITY_CONTROL, force=0)

        for _ in range(10):
            p.stepSimulation()

        self.current_step = 0
        return self._get_obs(), {}

    def _get_obs(self):
        # Example: [pole_angle, pole_vel, base_x, base_y, base_yaw, left_wheel_vel, right_wheel_vel]
        joint_states = [p.getJointState(self.robot_id, i) for i in range(2)]  # wheels
        pole_state = p.getLinkState(self.robot_id, 2, computeLinkVelocity=1)  # pole
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, _ = p.getBaseVelocity(self.robot_id)

        # Example observation: adjust indices as per your URDF
        obs = np.array([
            pole_state[1][1],  # pole pitch angle (y axis)
            pole_state[6][1],  # pole pitch velocity
            base_pos[0],       # base x
            base_pos[1],       # base y
            base_orn[2],       # base yaw
            joint_states[0][1],  # left wheel velocity
            joint_states[1][1],  # right wheel velocity
        ], dtype=np.float32)
        return obs

    def _get_reward(self, obs):
        # Reward upright pole and forward motion, penalize large angles
        pole_angle = obs[0]
        return -abs(pole_angle)

    def _is_done(self, obs):
        pole_angle = obs[0]
        base_x = obs[2]
        # Terminate if pole falls or base moves too far
        return abs(pole_angle) > 0.5 or abs(base_x) > 2.0

    def close(self):
        p.disconnect()

class AssistiveWalkerDiscreteEnv(AssistiveWalkerBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.action_space = spaces.Discrete(3)  # 0: left, 1: right, 2: stop
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        # Simple torque control for wheels
        torque = 10.0
        if action == 0:  # left
            left, right = -torque, torque
        elif action == 1:  # right
            left, right = torque, -torque
        else:  # stop
            left, right = 0.0, 0.0

        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=left)
        p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=right)

        p.stepSimulation()
        self.current_step += 1

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False
        return obs, reward, terminated, truncated, {}

class AssistiveWalkerContinuousEnv(AssistiveWalkerBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.action_space = spaces.Box(low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf], dtype=np.float32),
            high=np.array([np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf], dtype=np.float32),
            dtype=np.float32
        )

    def step(self, action):
        action = np.array(action, dtype=float).flatten()
        left_torque = float(action[0])
        right_torque = float(action[1])

        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=left_torque)
        p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=right_torque)

        p.stepSimulation()
        self.current_step += 1

        obs = self._get_obs()
        reward = self._get_reward(obs)
        terminated = self._is_done(obs)
        truncated = False
        return obs, reward, terminated, truncated, {}
