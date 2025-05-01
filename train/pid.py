import numpy as np
import matplotlib.pyplot as plt
import time
import gymnasium as gym
import os
import sys

# Fix for _file_ typo and import your custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from environments.cartpole import CartPoleContinuousEnv  # your custom env
import pybullet as p

class PIDController:
    def __init__ (self, Kp, Ki, Kd):  # Fixed __init_
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.prev_error = 0
        self.integral = 0

    def compute(self, error, dt):
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)
        self.prev_error = error
        return output

# Initialize environment
env = CartPoleContinuousEnv(render_mode=True)
obs, _ = env.reset()

# PID controller
pid = PIDController(Kp=100, Ki=0.1, Kd=10)

# Simulation parameters
max_steps = 1000
dt = env.time_step

# Logging data for plotting
time_log = []
pole_angle_log = []
cart_position_log = []

for step in range(max_steps):
    pole_angle = obs[0]        # theta (radians)
    cart_position = obs[1]     # x

    error = -pole_angle
    torque = pid.compute(error, dt)

    # Apply torque
    p.setJointMotorControl2(env.cartpole_id, 0, p.TORQUE_CONTROL, force=torque)
    p.setJointMotorControl2(env.cartpole_id, 1, p.TORQUE_CONTROL, force=0)

    p.stepSimulation()
    time.sleep(dt)

    obs = env._get_obs()
    done = env._is_done(obs)

    # Log data
    time_log.append(step * dt)
    pole_angle_log.append(pole_angle)
    cart_position_log.append(cart_position)

    if done:
        print(f"Episode ended at step {step}")
        break

env.close()

# Plotting
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(time_log, pole_angle_log, label="Pole Angle (rad)")
plt.axhline(0, color='gray', linestyle='--')
plt.title("Pole Angle Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Angle (rad)")
plt.grid(True)
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(time_log, cart_position_log, label="Cart Position (m)", color='orange')
plt.axhline(0, color='gray', linestyle='--')
plt.title("Cart Position Over Time")
plt.xlabel("Time (s)")
plt.ylabel("Position (m)")
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



# import numpy as np
# import matplotlib.pyplot as plt
# import time
# import gymnasium as gym
# import os
# import sys
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from environments.walker import WalkerBalanceContinuousEnv  # your env file

# # PID controller
# class PIDController:
#     def __init__(self, kp, ki, kd, setpoint=0.0):

#         self.kp = kp
#         self.ki = ki
#         self.kd = kd
#         self.setpoint = setpoint
#         self.integral = 0.0
#         self.prev_error = 0.0

#     def compute(self, measurement, dt):
#         error = self.setpoint - measurement
#         self.integral += error * dt
#         derivative = (error - self.prev_error) / dt if dt > 0 else 0.0
#         self.prev_error = error
#         output = self.kp * error + self.ki * self.integral + self.kd * derivative
#         return output

# # PID simulation and plotting
# def run_pid_with_plot(render_mode=False, duration_sec=10):
#     env = WalkerBalanceContinuousEnv(render_mode=render_mode, test_mode=False)
#     obs, _ = env.reset()

#     # Initialize PID for pole angle
#     pid = PIDController(kp=50.0, ki=1.0, kd=2.5)
#     timestep = env.time_step

#     # Logs
#     time_log = []
#     angle_log = []
#     control_log = []
#     reward_log = []

#     total_reward = 0.0
#     t = 0.0

#     for step in range(int(duration_sec / timestep)):
#         pole_angle = obs[0]
#         control = pid.compute(pole_angle, timestep)

#         action = np.array([-control, -control], dtype=np.float32)  # Same torque both wheels
#         obs, reward, terminated, truncated, info = env.step(action)

#         # Log
#         time_log.append(t)
#         angle_log.append(pole_angle)
#         control_log.append(control)
#         reward_log.append(reward)

#         total_reward += reward
#         t += timestep

#         if render_mode:
#             time.sleep(timestep)

#         if terminated:
#             print(f"Terminated at step {step}")
#             break

#     env.close()
#     print(f"Total Reward: {total_reward:.2f}")

#     # Plotting
#     plt.figure(figsize=(12, 6))
#     plt.subplot(2, 1, 1)
#     plt.plot(time_log, angle_log, label="Pole Angle (rad)")
#     plt.axhline(y=0.5, color='r', linestyle='--', label="Limit")
#     plt.axhline(y=-0.5, color='r', linestyle='--')
#     plt.ylabel("Pole Angle")
#     plt.legend()

#     plt.subplot(2, 1, 2)
#     plt.plot(time_log, control_log, label="Control Torque (Nm)", color='orange')
#     plt.ylabel("Torque")
#     plt.xlabel("Time (s)")
#     plt.legend()

#     plt.tight_layout()
#     plt.show()


# if __name__ == "__main__":
#     run_pid_with_plot(render_mode=False)