# import pybullet as p
# import pybullet_data
# import time

# # Start PyBullet GUI with specified resolution
# physicsClient = p.connect(p.GUI, options="--width=1280 --height=720")

# # Check if the connection was successful
# if physicsClient < 0:
#     print("Failed to connect to physics server!")
# else:
#     print("Connected to physics server.")

# # Set gravity
# p.setGravity(0, 0, -9.8)

# # Point to pybullet's internal data path
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Load a flat plane
# planeId = p.loadURDF("plane.urdf")

# # Load the built-in CartPole URDF (pendulum on a cart)
# cartpoleId = p.loadURDF("cartpole.urdf", basePosition=[0, 0, 0.1])

# # Run simulation for a while
# for i in range(1000):
#     p.stepSimulation()
#     time.sleep(1. / 240.)

# # Clean up
# p.disconnect()


# import pybullet as p
# import pybullet_data
# import time

# # Start PyBullet GUI
# physicsClient = p.connect(p.GUI)

# # Set gravity
# p.setGravity(0, 0, -9.8)

# # Point to pybullet's internal data path
# p.setAdditionalSearchPath(pybullet_data.getDataPath())

# # Load a flat plane
# planeId = p.loadURDF("plane.urdf")

# # Load the built-in CartPole URDF (pendulum on a cart)
# cartpoleId = p.loadURDF("cartpole.urdf", basePosition=[0, 0, 0.1])

# # Run simulation with an infinite loop to keep the GUI open
# while True:
#     p.stepSimulation()  # Perform one simulation step
#     time.sleep(1. / 240.)  # Control the simulation speed (in this case, 240 Hz)

# import pybullet as p
# import pybullet_data
# import time

# physicsClient = p.connect(p.GUI, options="--renderer=TinyRenderer")
# p.setAdditionalSearchPath(pybullet_data.getDataPath())
# p.setGravity(0, 0, -9.8)
# planeId = p.loadURDF("plane.urdf")
# cartpoleId = p.loadURDF("cartpole.urdf", basePosition=[0, 0, 0.1])

# while True:
#     p.stepSimulation()
#     time.sleep(1. / 240.)


import pybullet as p
import pybullet_data
import time
import os

# Start PyBullet GUI with specified resolution
physicsClient = p.connect(p.GUI, options="--width=1280 --height=720")

# Check if the connection was successful
if physicsClient < 0:
    print("Failed to connect to physics server!")
else:
    print("Connected to physics server.")

# Set gravity
p.setGravity(0, 0, -9.8)

# Point to pybullet's internal data path
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load a flat plane
planeId = p.loadURDF("plane.urdf")

# Define the correct path to your custom URDF
urdf_path = os.path.join(os.path.dirname(__file__), "urdf files", "chasis1.urdf")

# Load the custom cartpole URDF (chasis1.urdf in the urdf files folder)
cartpoleId = p.loadURDF(urdf_path, basePosition=[0, 0, 0.1])

# Run simulation for a while
for i in range(60*240):
    p.stepSimulation()
    time.sleep(1. / 240.)

# Clean up
p.disconnect()
