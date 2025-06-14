import gymnasium
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os

# Optional: Import wandb if you want to log IMU data
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

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

        # Set IMU link index (update this according to your URDF)
        self.imu_link_index = 3  # CHANGE THIS if your imu_link index is different
        self.prev_lin_vel = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at {urdf_path}")

        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0.15])

        # Initial joint indices for the wheels
        init_left_wheel = 0
        init_right_wheel = 1

        # Set friction before stepping simulation
        p.changeDynamics(self.robot_id, init_left_wheel, lateralFriction=1.2)
        p.changeDynamics(self.robot_id, init_right_wheel, lateralFriction=1.2)

        # Initialize pole and wheel angles
        init_pole_angle = np.random.uniform(-0.2, 0.2)
        left_wheel_angle = 0
        right_wheel_angle = 0

        p.resetJointState(self.robot_id, init_left_wheel, left_wheel_angle, 0)
        p.resetJointState(self.robot_id, init_right_wheel, right_wheel_angle, 0)
        p.resetJointState(self.robot_id, 2, init_pole_angle, 0)  # Assuming 2 is pole joint

        # Disable default motors for wheels
        for joint in [init_left_wheel, init_right_wheel]:
            p.setJointMotorControl2(self.robot_id, joint, p.VELOCITY_CONTROL, force=0)

        # Let system settle
        for _ in range(10):
            p.stepSimulation()

        self.current_step = 0
        self.prev_lin_vel = np.zeros(3)
        return self._get_obs(), {}


    def _get_obs(self):
        # Wheel joint states
        joint_states = [p.getJointState(self.robot_id, i) for i in range(2)]  # wheels
        # Pole state
        pole_state = p.getLinkState(self.robot_id, 2, computeLinkVelocity=1)  # pole
        base_pos, base_orn = p.getBasePositionAndOrientation(self.robot_id)
        base_vel, _ = p.getBaseVelocity(self.robot_id)

        # Main robot observation (as before)
        obs = np.array([
            pole_state[1][1],  # pole pitch angle (y axis)
            pole_state[6][1],  # pole pitch velocity
            base_pos[0],       # base x
            base_pos[1],       # base y
            base_orn[2],       # base yaw
            joint_states[0][1],  # left wheel velocity
            joint_states[1][1],  # right wheel velocity
        ], dtype=np.float32)

        # --- IMU Sensor Integration ---
        imu_state = p.getLinkState(self.robot_id, self.imu_link_index, computeLinkVelocity=1)
        imu_quat = imu_state[1]
        imu_lin_vel = np.array(imu_state[6])
        imu_ang_vel = np.array(imu_state[7])
        lin_acc = (imu_lin_vel - self.prev_lin_vel) / self.time_step
        self.prev_lin_vel = imu_lin_vel
        imu_euler = p.getEulerFromQuaternion(imu_quat)
        imu_obs = np.concatenate((np.array(imu_euler), imu_ang_vel, lin_acc))
        imu_state = p.getLinkState(self.robot_id, self.imu_link_index, computeLinkVelocity=1)
        if imu_state is None:
            raise RuntimeError(f"IMU link index {self.imu_link_index} not found. Check URDF and link indices.")
        imu_quat = imu_state[1]

        # Concatenate IMU data to observation
        obs = np.concatenate((obs, imu_obs))
        return obs

    def _get_reward(self, obs):
        pole_angle = obs[0]
        base_x = obs[2]
        left_wheel_vel = obs[5]
        right_wheel_vel = obs[6]

        fallen = abs(pole_angle) > 0.8

        angle_penalty = 0.5 * abs(pole_angle)
        base_penalty = 0.05 * abs(base_x)
        vel_penalty = 0.002 * (left_wheel_vel**2 + right_wheel_vel**2)
        vel_penalty = min(vel_penalty, 1.0)  # cap it

        reward = 1.0 - angle_penalty - base_penalty - vel_penalty

        if not fallen:
            reward += 2.0
        else:
            reward -= 20.0

        return reward


        return reward

    def _is_done(self, obs):
        pole_angle = obs[0]
        base_x = obs[2]
        return abs(pole_angle) > 0.3 or abs(base_x) > 2.0

    def close(self):
        p.disconnect()

class AssistiveWalkerDiscreteEnv(AssistiveWalkerBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        # 7 original obs + 9 IMU = 16
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,   # original
                 -np.pi, -np.pi, -np.pi,   # IMU orientation (roll, pitch, yaw)
                 -np.inf, -np.inf, -np.inf,  # IMU angular velocity
                 -np.inf, -np.inf, -np.inf], # IMU linear acceleration
                dtype=np.float32),
            high=np.array(
                [np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                 np.pi, np.pi, np.pi,
                 np.inf, np.inf, np.inf,
                 np.inf, np.inf, np.inf],
                dtype=np.float32
            ),
            dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def step(self, action):
        torque = 2
        if action == 0:
            left, right = -torque, torque
        elif action == 1:
            left, right = torque, -torque
        else:
            left, right = 0.0, 0.0

        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=left)
        p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=right)
        p.stepSimulation()
        self.current_step += 1

        obs_full = self._get_obs()
        obs_agent = obs_full[:7]
        reward = self._get_reward(obs_full)
        terminated = self._is_done(obs_full)
        truncated = False

        # Log IMU data to WandB (if enabled)
        if WANDB_AVAILABLE:
            wandb.log({
                "imu_roll": obs_full[7],
                "imu_pitch": obs_full[8],
                "imu_yaw": obs_full[9],
                "imu_ang_vel_x": obs_full[10],
                "imu_ang_vel_y": obs_full[11],
                "imu_ang_vel_z": obs_full[12],
                "imu_lin_acc_x": obs_full[13],
                "imu_lin_acc_y": obs_full[14],
                "imu_lin_acc_z": obs_full[15],
                "step": self.current_step
            })

        info = {"imu": obs_full[7:]}
        return obs_agent, reward, terminated, truncated, info

class AssistiveWalkerContinuousEnv(AssistiveWalkerBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode=render_mode, test_mode=test_mode)
        self.observation_space = spaces.Box(
            low=np.array(
                [-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
                 -np.pi, -np.pi, -np.pi,
                 -np.inf, -np.inf, -np.inf,
                 -np.inf, -np.inf, -np.inf],
                dtype=np.float32),
            high=np.array(
                [np.pi, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                 np.pi, np.pi, np.pi,
                 np.inf, np.inf, np.inf,
                 np.inf, np.inf, np.inf],
                dtype=np.float32
            ),
            dtype=np.float32
        )
        self.action_space = spaces.Box(low=np.array([-5, -5]), high=np.array([5, 5]), dtype=np.float32)

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
        

        # Example: Log IMU data to WandB (if enabled)
        if WANDB_AVAILABLE:
            wandb.log({
                "imu_roll": obs[7],
                "imu_pitch": obs[8],
                "imu_yaw": obs[9],
                "imu_ang_vel_x": obs[10],
                "imu_ang_vel_y": obs[11],
                "imu_ang_vel_z": obs[12],
                "imu_lin_acc_x": obs[13],
                "imu_lin_acc_y": obs[14],
                "imu_lin_acc_z": obs[15],
                "step": self.current_step
            })

        return obs, reward, terminated, truncated, {}
