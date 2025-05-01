import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pybullet as p
import pybullet_data
import os
import time


class WalkerBalanceBaseEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"],"render_fps": 60}
    gui_connected = False

    def __init__(self, render_mode=None, test_mode=False):
        super().__init__()
        self.render_mode = render_mode
        self.test_mode = test_mode

        # Connect to PyBullet
        if self.render_mode:
            if not WalkerBalanceBaseEnv.gui_connected:
                p.connect(p.GUI)
                p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
                WalkerBalanceBaseEnv.gui_connected = True
            else:
                p.connect(p.DIRECT)
        else:
            p.connect(p.DIRECT)

        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        self.time_step = 1.0 / 240.0
        p.setTimeStep(self.time_step)
        # Track last disturbance torque
        self.last_disturbance = None
        self.robot_id = None
        self.pole_joint_index = 2  # index of pole joint

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        p.resetSimulation()
        p.setGravity(0, 0, -9.81)
        p.loadURDF("plane.urdf")

        # Load robot URDF
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
        urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
        if not os.path.exists(urdf_path):
            raise FileNotFoundError(f"URDF file not found at {urdf_path}")
        self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

        # Disable all motors (wheels and pole)
        for j in range(p.getNumJoints(self.robot_id)):
            p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

        # step a few times to settle
        for _ in range(10):
            p.stepSimulation()

        self.current_step = 0
        if self.test_mode:
            self.disturbance_step = np.random.randint(100, 300)

        return self._get_obs(), {}

    def disturb_pole(self):
        # pick a random torque
        torque = np.random.uniform(-2.0, 2.0)
        # apply it
        p.setJointMotorControl2(
            self.robot_id,
            self.pole_joint_index,
            p.TORQUE_CONTROL,
            force=torque
        )
        # record it
        self.last_disturbance = torque

    def render(self):
        # Only render when human mode
        if not self.render_mode:
            return
        # single-step rendering for smooth GUI
        p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
        p.stepSimulation()
        time.sleep(self.time_step)
        # Mouse click disturbance: index 0 is left click, state bit KEY_WAS_TRIGGERED is e[4]
        for e in p.getMouseEvents():
            # e = (eventType, buttonIndex, x, y, buttonState)
            if e[1] == 0 and (e[4] & p.KEY_WAS_TRIGGERED):
                # shoot a ray from camera center
                _, _, view_matrix, proj_matrix, _, _ = p.getDebugVisualizerCamera()
                ray_start, ray_end = p.computeViewRay(view_matrix, proj_matrix, 0.5, 0.5)
                hit = p.rayTest(ray_start, ray_end)[0]
                print("[Render] mouse event:", e)
                if hit[0] >= 0:
                    # p.applyExternalForce(
                    #     objectUniqueId=hit[0],
                    #     linkIndex=-1,
                    #     forceObj=(0, 0, 50),
                    #     posObj=hit[3],
                    #     flags=p.WORLD_FRAME
                    # )
                    self.disturb_pole()
                    print(f"[Render] Applied random torque {self.last_disturbance:.3f} Nm to pole")
        time.sleep(self.time_step)

    def _get_obs(self):
        js0 = p.getJointState(self.robot_id, 0)
        js1 = p.getJointState(self.robot_id, 1)
        jsp = p.getJointState(self.robot_id, self.pole_joint_index)
        return np.array([
            jsp[0], jsp[1],
            js0[0], js0[1],
            js1[0], js1[1]
        ], dtype=np.float32)

    def _get_reward(self, obs):
        angle, ang_vel = obs[0], obs[1]
        return -abs(angle) - 0.1 * abs(ang_vel)

    def _is_done(self, obs):
        return abs(obs[0]) > 0.5

    def close(self):
        p.disconnect()


class WalkerBalanceDiscreteEnv(WalkerBalanceBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode, test_mode)
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([ np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]),
            dtype=np.float32
        )

    def step(self, action):
        # map action {0:left,1:stop,2:right} to torque
        mapping = {0: -2.0, 1: 0.0, 2: 2.0}
        force = mapping.get(action, 0.0)
        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=force)
        p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=force)
        p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=0)
        p.stepSimulation()
        obs = self._get_obs()
        return obs, self._get_reward(obs), self._is_done(obs), False, {}


class WalkerBalanceContinuousEnv(WalkerBalanceBaseEnv):
    def __init__(self, render_mode=False, test_mode=False):
        super().__init__(render_mode, test_mode)
        self.action_space = spaces.Box(
            low=np.array([-5., -5.]),
            high=np.array([ 5.,  5.]),
            dtype=np.float32
        )
        self.observation_space = spaces.Box(
            low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
            high=np.array([ np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]),
            dtype=np.float32
        )

    def step(self, action):
        lt, rt = map(float, np.array(action).reshape(-1))
        p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=lt)
        p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=rt)
        p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=0)
        p.stepSimulation()
        obs = self._get_obs()
        done = self._is_done(obs)
        return obs, self._get_reward(obs), done, False, {}


# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pybullet as p
# import pybullet_data
# import os
# import time
# import random


# class WalkerBalanceBaseEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}
#     gui_connected = False

#     def __init__(self, render_mode=None, test_mode=False):
#         super().__init__()
#         self.render_mode = render_mode
#         self.test_mode = test_mode

#         # Connect to PyBullet
#         if self.render_mode:
#             if not WalkerBalanceBaseEnv.gui_connected:
#                 p.connect(p.GUI)
#                 p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
#                 WalkerBalanceBaseEnv.gui_connected = True
#             else:
#                 p.connect(p.DIRECT)
#         else:
#             p.connect(p.DIRECT)

#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         self.time_step = 1.0 / 240.0
#         p.setTimeStep(self.time_step)

#         self.robot_id = None
#         self.pole_joint_index = 2  # index of pole joint
#         self.current_step = 0
#         self.test_mode = test_mode

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.81)
#         p.loadURDF("plane.urdf")

#         # Load robot URDF
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#         urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF file not found at {urdf_path}")
#         self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

#         # Disable all motors
#         for j in range(p.getNumJoints(self.robot_id)):
#             p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

#         for _ in range(10):
#             p.stepSimulation()

#         self.current_step = 0
#         if self.test_mode:
#             self.disturbance_step = np.random.randint(100, 300)
#         return self._get_obs(), {}

#     def disturb_pole(self):
#         torque = np.random.uniform(-5.0, 5.0)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=torque)

#     def render(self):
#         if not self.render_mode:
#             return
#         p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
#         p.stepSimulation()

#         # Mouse click disturbance
#         for e in p.getMouseEvents():
#             if e[0] == p.MOUSE_BUTTON_LEFT and e[3] & p.KEY_WAS_TRIGGERED:
#                 _, _, view_matrix, proj_matrix, _, _ = p.getDebugVisualizerCamera()
#                 ray_start, ray_end = p.computeViewRay(view_matrix, proj_matrix, 0.5, 0.5)
#                 hit = p.rayTest(ray_start, ray_end)[0]
#                 if hit[0] >= 0:
#                     p.applyExternalForce(
#                         objectUniqueId=hit[0],
#                         linkIndex=-1,
#                         forceObj=(0, 0, 50),
#                         posObj=hit[3],
#                         flags=p.WORLD_FRAME
#                     )
#         time.sleep(self.time_step)

#     def _get_obs(self):
#         js0 = p.getJointState(self.robot_id, 0)
#         js1 = p.getJointState(self.robot_id, 1)
#         jsp = p.getJointState(self.robot_id, self.pole_joint_index)
#         return np.array([jsp[0], jsp[1], js0[0], js0[1], js1[0], js1[1]], dtype=np.float32)

#     def _get_reward(self, obs):
#         return -abs(obs[0]) - 0.1 * abs(obs[1])

#     def _is_done(self, obs):
#         return abs(obs[0]) > 0.5

#     def close(self):
#         p.disconnect()


# class WalkerBalanceDiscreteEnv(WalkerBalanceBaseEnv):
#     def __init__(self, render_mode=False, test_mode=False):
#         super().__init__(render_mode, test_mode)
#         self.action_space = spaces.Discrete(3)
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([ np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]),
#             dtype=np.float32
#         )

#     def step(self, action):
#         force = {-1: -2.0, 0:0.0, 1:2.0}.get(action-1, 0.0)
#         p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=force)
#         p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=force)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=0)
#         p.stepSimulation()
#         obs = self._get_obs()
#         return obs, self._get_reward(obs), self._is_done(obs), False, {}


# class WalkerBalanceContinuousEnv(WalkerBalanceBaseEnv):
#     def __init__(self, render_mode=False, test_mode=False):
#         super().__init__(render_mode, test_mode)
#         self.action_space = spaces.Box(
#             low=np.array([-5., -5.]),
#             high=np.array([ 5.,  5.]),
#             dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([ np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]),
#             dtype=np.float32
#         )

#     def step(self, action):
#         lt, rt = map(float, np.array(action).flatten())
#         p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=lt)
#         p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=rt)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=0)
#         p.stepSimulation()
#         obs = self._get_obs()
#         done = self._is_done(obs)
#         return obs, self._get_reward(obs), done, False, {}





# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pybullet as p
# import pybullet_data
# import os
# import time
# import random


# class WalkerBalanceBaseEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}
#     gui_connected = False

#     def __init__(self, render_mode=None, test_mode=False):
#         super().__init__()
#         self.render_mode = render_mode
#         self.test_mode = test_mode

#         if self.render_mode:
#             if not WalkerBalanceBaseEnv.gui_connected:
#                 p.connect(p.GUI)
#                 p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
#                 WalkerBalanceBaseEnv.gui_connected = True
#             else:
#                 p.connect(p.DIRECT)
#         else:
#             p.connect(p.DIRECT)

#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         self.time_step = 1.0 / 240.0
#         p.setTimeStep(self.time_step)

#         self.robot_id = None
#         self.pole_joint_index = 2  # assuming index 2 is pole
#         self.current_step = 0
#         self.disturbance_step = None
#         self.disturbed = False
#         self.steps_after_disturbance = 0
#         self.reward_after_disturbance = 0.0

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.81)
#         p.loadURDF("plane.urdf")

#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#         urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF file not found at {urdf_path}")

#         self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

#         # Disable all motors (wheels and pole)
#         for j in range(p.getNumJoints(self.robot_id)):
#             p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

#         # Now the pole can be moved freely with the mouse (if motor is disabled)
#         p.setJointMotorControl2(
#             bodyUniqueId=self.robot_id,
#             jointIndex=self.pole_joint_index,
#             controlMode=p.VELOCITY_CONTROL,
#             force=0
#         )

#         for _ in range(10):
#             p.stepSimulation()

#         self.current_step = 0
#         self.disturbed = False
#         self.steps_after_disturbance = 0
#         self.reward_after_disturbance = 0.0

#         if self.test_mode:
#             self.disturbance_step = np.random.randint(100, 300)

#         return self._get_obs(), {}

#     def disturb_pole(self):
#         random_torque = np.random.uniform(-5.0, 5.0)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=random_torque)

#     def render(self):
#         if self.render_mode:
#             p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
#             p.stepSimulation()

#             mouse_events = p.getMouseEvents()
#             for e in mouse_events:
#                 if e[0] == p.MOUSE_BUTTON_LEFT and e[3] & p.KEY_WAS_TRIGGERED:
#                     width, height, view_matrix, proj_matrix, _, _ = p.getDebugVisualizerCamera()
#                     ray_start, ray_end = p.computeViewRay(view_matrix, proj_matrix, 0.5, 0.5)
#                     hit = p.rayTest(ray_start, ray_end)[0]

#                     if hit[0] >= 0:
#                         print("Mouse click hit object:", hit[0], "at position", hit[3])
#                         p.applyExternalForce(
#                             objectUniqueId=hit[0],
#                             linkIndex=-1,
#                             forceObj=(0, 0, 50),
#                             posObj=hit[3],
#                             flags=p.WORLD_FRAME
#                         )

#             time.sleep(self.time_step)

# # class WalkerBalanceBaseEnv(gym.Env):
# #     metadata = {"render_modes": ["human", "rgb_array"]}
# #     gui_connected = False

# #     def __init__(self, render_mode=None, test_mode=False):
# #         super().__init__()
# #         self.render_mode = render_mode
# #         self.test_mode = test_mode

# #         if self.render_mode:
# #             if not WalkerBalanceBaseEnv.gui_connected:
# #                 p.connect(p.GUI)
# #                 p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
# #                 WalkerBalanceBaseEnv.gui_connected = True
# #             else:
# #                 p.connect(p.DIRECT)
# #         else:
# #             p.connect(p.DIRECT)

# #         p.setAdditionalSearchPath(pybullet_data.getDataPath())
# #         self.time_step = 1.0 / 240.0
# #         p.setTimeStep(self.time_step)

# #         self.robot_id = None
# #         self.current_step = 0
# #         self.disturbance_step = None
# #         self.disturbed = False
# #         self.steps_after_disturbance = 0
# #         self.reward_after_disturbance = 0.0

# #     def reset(self, seed=None, options=None):
# #         super().reset(seed=seed)
# #         p.resetSimulation()
# #         p.setGravity(0, 0, -9.81)
# #         p.loadURDF("plane.urdf")

# #         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
# #         urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
# #         if not os.path.exists(urdf_path):
# #             raise FileNotFoundError(f"URDF file not found at {urdf_path}")

# #         self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

# #         for j in range(p.getNumJoints(self.robot_id)):
# #             p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

# #         for _ in range(10):
# #             p.stepSimulation()

# #         self.current_step = 0
# #         self.disturbed = False
# #         self.steps_after_disturbance = 0
# #         self.reward_after_disturbance = 0.0

# #         if self.test_mode:
# #             self.disturbance_step = np.random.randint(100, 300)

# #         if self.render_mode:
# #             p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)

# #         return self._get_obs(), {}

# #     def disturb_pole(self):
# #         random_torque = np.random.uniform(-5.0, 5.0)
# #         p.setJointMotorControl2(self.robot_id, 2, p.TORQUE_CONTROL, force=random_torque)

# #     def render(self):
# #         if self.render_mode:
# #             # Let PyBullet render the scene
# #             p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
# #             p.stepSimulation()

# #             # Check mouse click events
# #             mouse_events = p.getMouseEvents()
# #             for e in mouse_events:
# #                 if e[0] == p.MOUSE_BUTTON_LEFT and e[3] & p.KEY_WAS_TRIGGERED:
# #                     width, height, view_matrix, proj_matrix, _, _ = p.getDebugVisualizerCamera()
# #                     ray_start, ray_end = p.computeViewRay(view_matrix, proj_matrix, 0.5, 0.5)
# #                     hit = p.rayTest(ray_start, ray_end)[0]

# #                     if hit[0] >= 0:
# #                         print("Mouse click hit object:", hit[0], "at position", hit[3])
# #                         p.applyExternalForce(
# #                             objectUniqueId=hit[0],
# #                             linkIndex=-1,
# #                             forceObj=(0, 0, 50),  # Upward force
# #                             posObj=hit[3],
# #                             flags=p.WORLD_FRAME
# #                         )

# #             time.sleep(self.time_step)


# #     def _get_obs(self):
# #         l_wheel = p.getJointState(self.robot_id, 0)
# #         r_wheel = p.getJointState(self.robot_id, 1)
# #         pole = p.getJointState(self.robot_id, 2)

# #         pole_angle = pole[0]
# #         pole_velocity = pole[1]
# #         l_wheel_position = l_wheel[0]
# #         l_wheel_velocity = l_wheel[1]
# #         r_wheel_position = r_wheel[0]
# #         r_wheel_velocity = r_wheel[1]

# #         return np.array([
# #             pole_angle,
# #             pole_velocity,
# #             l_wheel_position,
# #             l_wheel_velocity,
# #             r_wheel_position,
# #             r_wheel_velocity
# #         ], dtype=np.float32)

# #     def _get_reward(self, obs):
# #         pole_angle = obs[0]
# #         pole_velocity = obs[1]
# #         return -abs(pole_angle) - 0.1 * abs(pole_velocity)

# #     def _is_done(self, obs):
# #         return abs(obs[0]) > 0.5

# #     def close(self):
# #         p.disconnect()


# class WalkerBalanceDiscreteEnv(WalkerBalanceBaseEnv):
#     def __init__(self, render_mode=False, test_mode=False):
#         super().__init__(render_mode=render_mode, test_mode=test_mode)
#         self.action_space = spaces.Discrete(3)
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([np.pi, np.inf, np.inf, np.inf, np.inf, np.inf]),
#             dtype=np.float32
#         )

#     def step(self, action):
#         force = 0.0
#         if action == 0:
#             force = -2.0
#         elif action == 1:
#             force = 2.0

#         for j in [0, 1]:
#             p.setJointMotorControl2(self.robot_id, j, p.TORQUE_CONTROL, force=force)
#         p.setJointMotorControl2(self.robot_id, 2, p.TORQUE_CONTROL, force=0.0)

#         p.stepSimulation()
#         self.current_step += 1

#         if self.test_mode and self.current_step == self.disturbance_step:
#             self.disturb_pole()

#         obs = self._get_obs()
#         reward = self._get_reward(obs)
#         done = self._is_done(obs)

#         return obs, reward, done, False, {}


# class WalkerBalanceContinuousEnv(WalkerBalanceBaseEnv):
#     def __init__(self, render_mode=False, test_mode=False):
#         super().__init__(render_mode=render_mode, test_mode=test_mode)
#         self.action_space = spaces.Box(
#             low=np.array([-5.0, -5.0]), high=np.array([5.0, 5.0]), dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([np.pi, np.inf, np.inf, np.inf, np.inf, np.inf]),
#             dtype=np.float32
#         )

#     def step(self, action):
#         action = np.array(action, dtype=np.float32).flatten()
#         left_torque, right_torque = float(action[0]), float(action[1])

#         p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=left_torque)
#         p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=right_torque)
#         p.setJointMotorControl2(self.robot_id, 2, p.TORQUE_CONTROL, force=0.0)

#         p.stepSimulation()
#         self.current_step += 1

#         if self.test_mode and self.current_step == self.disturbance_step:
#             self.disturb_pole()

#         obs = self._get_obs()
#         reward = self._get_reward(obs)
#         terminated = self._is_done(obs)

#         if self.test_mode and self.current_step >= self.disturbance_step:
#             self.disturbed = True
#         if self.disturbed:
#             self.steps_after_disturbance += 1
#             self.reward_after_disturbance += reward

#         return obs, reward, terminated, False, {}


# import gymnasium as gym
# from gymnasium import spaces
# import numpy as np
# import pybullet as p
# import pybullet_data
# import os
# import time
# import random


# class WalkerBalanceBaseEnv(gym.Env):
#     metadata = {"render_modes": ["human", "rgb_array"]}
#     gui_connected = False

#     def __init__(self, render_mode=None, test_mode=False):
#         super().__init__()
#         self.render_mode = render_mode
#         self.test_mode = test_mode

#         # Connect to PyBullet
#         if self.render_mode:
#             if not WalkerBalanceBaseEnv.gui_connected:
#                 p.connect(p.GUI)
#                 p.configureDebugVisualizer(p.COV_ENABLE_MOUSE_PICKING, 1)
#                 WalkerBalanceBaseEnv.gui_connected = True
#             else:
#                 p.connect(p.DIRECT)
#         else:
#             p.connect(p.DIRECT)

#         p.setAdditionalSearchPath(pybullet_data.getDataPath())
#         self.time_step = 1.0 / 240.0
#         p.setTimeStep(self.time_step)

#         self.robot_id = None
#         self.pole_joint_index = 2  # index of pole joint
#         self.current_step = 0
#         self.test_mode = test_mode

#     def reset(self, seed=None, options=None):
#         super().reset(seed=seed)
#         p.resetSimulation()
#         p.setGravity(0, 0, -9.81)
#         p.loadURDF("plane.urdf")

#         # Load robot URDF
#         project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
#         urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
#         if not os.path.exists(urdf_path):
#             raise FileNotFoundError(f"URDF file not found at {urdf_path}")
#         self.robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])

#         # Disable all motors
#         for j in range(p.getNumJoints(self.robot_id)):
#             p.setJointMotorControl2(self.robot_id, j, p.VELOCITY_CONTROL, force=0)

#         for _ in range(10):
#             p.stepSimulation()

#         self.current_step = 0
#         if self.test_mode:
#             self.disturbance_step = np.random.randint(100, 300)
#         return self._get_obs(), {}

#     def disturb_pole(self):
#         torque = np.random.uniform(-5.0, 5.0)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=torque)

#     def render(self):
#         if not self.render_mode:
#             return
#         p.configureDebugVisualizer(p.COV_ENABLE_SINGLE_STEP_RENDERING, 1)
#         p.stepSimulation()

#         # Mouse click disturbance
#         for e in p.getMouseEvents():
#             if e[0] == p.MOUSE_BUTTON_LEFT and e[3] & p.KEY_WAS_TRIGGERED:
#                 _, _, view_matrix, proj_matrix, _, _ = p.getDebugVisualizerCamera()
#                 ray_start, ray_end = p.computeViewRay(view_matrix, proj_matrix, 0.5, 0.5)
#                 hit = p.rayTest(ray_start, ray_end)[0]
#                 if hit[0] >= 0:
#                     p.applyExternalForce(
#                         objectUniqueId=hit[0],
#                         linkIndex=-1,
#                         forceObj=(0, 0, 50),
#                         posObj=hit[3],
#                         flags=p.WORLD_FRAME
#                     )
#         time.sleep(self.time_step)

#     def _get_obs(self):
#         js0 = p.getJointState(self.robot_id, 0)
#         js1 = p.getJointState(self.robot_id, 1)
#         jsp = p.getJointState(self.robot_id, self.pole_joint_index)
#         return np.array([jsp[0], jsp[1], js0[0], js0[1], js1[0], js1[1]], dtype=np.float32)

#     def _get_reward(self, obs):
#         return -abs(obs[0]) - 0.1 * abs(obs[1])

#     def _is_done(self, obs):
#         return abs(obs[0]) > 0.5

#     def close(self):
#         p.disconnect()


# class WalkerBalanceDiscreteEnv(WalkerBalanceBaseEnv):
#     def __init__(self, render_mode=False, test_mode=False):
#         super().__init__(render_mode, test_mode)
#         self.action_space = spaces.Discrete(3)
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([ np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]),
#             dtype=np.float32
#         )

#     def step(self, action):
#         force = {-1: -2.0, 0:0.0, 1:2.0}.get(action-1, 0.0)
#         p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=force)
#         p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=force)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=0)
#         p.stepSimulation()
#         obs = self._get_obs()
#         return obs, self._get_reward(obs), self._is_done(obs), False, {}


# class WalkerBalanceContinuousEnv(WalkerBalanceBaseEnv):
#     def __init__(self, render_mode=False, test_mode=False):
#         super().__init__(render_mode, test_mode)
#         self.action_space = spaces.Box(
#             low=np.array([-5., -5.]),
#             high=np.array([ 5.,  5.]),
#             dtype=np.float32
#         )
#         self.observation_space = spaces.Box(
#             low=np.array([-np.pi, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf]),
#             high=np.array([ np.pi,  np.inf,  np.inf,  np.inf,  np.inf,  np.inf]),
#             dtype=np.float32
#         )

#     def step(self, action):
#         lt, rt = map(float, np.array(action).flatten())
#         p.setJointMotorControl2(self.robot_id, 0, p.TORQUE_CONTROL, force=lt)
#         p.setJointMotorControl2(self.robot_id, 1, p.TORQUE_CONTROL, force=rt)
#         p.setJointMotorControl2(self.robot_id, self.pole_joint_index, p.TORQUE_CONTROL, force=0)
#         p.stepSimulation()
#         obs = self._get_obs()
#         done = self._is_done(obs)
#         return obs, self._get_reward(obs), done, False, {}
