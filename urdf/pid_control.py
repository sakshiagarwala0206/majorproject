import pybullet as p
import time
import numpy as np
import matplotlib.pyplot as plt

# PID Controller Class
class PIDController:
    def __init__(self, kp, ki, kd):
        self.kp = kp
        self.ki = ki
        self.kd = kd
        self.previous_error = 0
        self.integral = 0

    def calculate(self, setpoint, measurement, dt):
        error = setpoint - measurement
        self.integral += error * dt
        derivative = (error - self.previous_error) / dt
        self.previous_error = error
        return self.kp * error + self.ki * self.integral + self.kd * derivative

# Initialize the simulation and load the URDF model
physicsClient = p.connect(p.GUI)
p.setGravity(0, 0, -9.81)

# Load your custom URDF model
robot = p.loadURDF("plane.urdf", basePosition=[0, 0, 1])

# Get joint indices
slider_joint = 0  # Change to your specific joint index for the slider
hinge_joint = 1  # Change to your specific joint index for the hinge (pole)

# Define PID controllers for the pole angle and cart position
pole_pid = PIDController(kp=100, ki=0.01, kd=5)  # Adjust PID gains for balancing the pole
cart_pid = PIDController(kp=10, ki=0.0, kd=2)   # Adjust PID gains for cart position

# Setpoint for the pole (target upright position) and cart (target position at the center)
target_angle = 0  # Upright angle (in radians)
target_position = 0  # Center position (in meters)

# Lists to collect data for plotting
time_list = []
pole_angle_list = []
pole_velocity_list = []
cart_position_list = []
cart_velocity_list = []

# Simulation loop
time_step = 0.01
for t in range(10000):
    # Step the simulation
    p.stepSimulation()
    time.sleep(time_step)

    # Get the current state of the joint (the angle and angular velocity of the pole)
    joint_state = p.getJointState(robot, hinge_joint)
    pole_angle = joint_state[0]      # Radians
    pole_velocity = joint_state[1]   # Angular velocity

    # Get the current position and velocity of the cart
    joint_state_cart = p.getJointState(robot, slider_joint)
    cart_position = joint_state_cart[0]  # Cart position (meters)
    cart_velocity = joint_state_cart[1]  # Cart velocity (m/s)

    # Calculate PID control for balancing the pole (angle)
    pole_torque = pole_pid.calculate(target_angle, pole_angle, time_step)

    # Calculate PID control for moving the cart to the center (position)
    cart_force = cart_pid.calculate(target_position, cart_position, time_step)

    # Apply the forces/torques to the joints
    # Apply the computed torque to the hinge joint (pole)
    p.setJointMotorControl2(
        bodyUniqueId=robot,
        jointIndex=hinge_joint,
        controlMode=p.TORQUE_CONTROL,
        force=pole_torque
    )

    # Apply the computed force to the slider joint (cart)
    p.setJointMotorControl2(
        bodyUniqueId=robot,
        jointIndex=slider_joint,
        controlMode=p.TORQUE_CONTROL,
        force=cart_force
    )

    # Collect data for plotting
    time_list.append(t * time_step)
    pole_angle_list.append(pole_angle)
    pole_velocity_list.append(pole_velocity)
    cart_position_list.append(cart_position)
    cart_velocity_list.append(cart_velocity)

# After the simulation, plot the results
plt.figure(figsize=(12, 8))

# Plot the pole angle
plt.subplot(2, 2, 1)
plt.plot(time_list, pole_angle_list, label="Pole Angle (radians)")
plt.title("Pole Angle vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Pole Angle (radians)")

# Plot the pole velocity
plt.subplot(2, 2, 2)
plt.plot(time_list, pole_velocity_list, label="Pole Velocity (rad/s)", color='r')
plt.title("Pole Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Pole Velocity (rad/s)")

# Plot the cart position
plt.subplot(2, 2, 3)
plt.plot(time_list, cart_position_list, label="Cart Position (m)", color='g')
plt.title("Cart Position vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Cart Position (m)")

# Plot the cart velocity
plt.subplot(2, 2, 4)
plt.plot(time_list, cart_velocity_list, label="Cart Velocity (m/s)", color='b')
plt.title("Cart Velocity vs Time")
plt.xlabel("Time (s)")
plt.ylabel("Cart Velocity (m/s)")

# Adjust layout and show plots
plt.tight_layout()
plt.show()
