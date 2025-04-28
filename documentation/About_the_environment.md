## CartPole Environment Documentation

This document describes the implementation of CartPole environments using the PyBullet physics engine. The environments are designed for use with the Gymnasium reinforcement learning framework and provide both discrete and continuous action spaces. The implementation includes a base environment class and specialized classes for each action space type, along with a disturbance capability for robustness testing.

### 1. Introduction

The CartPole environment is a classic control problem in which a pole is attached to a cart moving along a horizontal track. The goal is to balance the pole upright by applying forces or torques to the cart. This implementation provides a realistic simulation of the CartPole system using PyBullet, a physics engine, and adheres to the Gymnasium interface.

### 2. Class Descriptions

The code consists of three primary classes:

* `CartPoleBaseEnv`: The base class that encapsulates the common functionality for all CartPole variations.
* `CartPoleDiscreteEnv`: An environment with a discrete action space, where the agent can apply a fixed amount of torque to the cart in either the left or right direction.
* `CartPoleContinuousEnv`: An environment with a continuous action space, where the agent can apply variable torques to both the cart and the pole.

### 3. Class Details

#### 3.1 `CartPoleBaseEnv`

The `CartPoleBaseEnv` class inherits from `gymnasium.Env` and provides the core simulation setup and logic.

##### 3.1.1 Attributes

* `metadata`: A dictionary defining the supported rendering modes: "human" for on-screen visualization and "rgb_array" for rendering to an RGB array.
* `gui_connected`: A class-level boolean flag to track if the PyBullet GUI has been connected. This ensures that the GUI is initialized only once.
* `render_mode`: The rendering mode for the environment (None, "human", or "rgb_array").
* `viewer`: The PyBullet viewer object (if rendering is enabled).
* `render_fps`: The frames per second for rendering (set to 60).
* `test_mode`: A boolean flag indicating whether the environment is in test mode.
* `time_step`: The simulation time step (set to 1/240 seconds).
* `cartpole_id`: The PyBullet ID of the CartPole robot.
* `current_step`: The current time step in the episode.
* `disturbance_step`: The time step at which a disturbance will be applied (in test mode).
* `disturbed`: A boolean flag indicating if a disturbance has been applied.
* `steps_after_disturbance`: The number of steps after a disturbance has been applied.
* `reward_after_disturbance`: The cumulative reward after a disturbance.

##### 3.1.2 Methods

* `__init__(self, render_mode=None, test_mode=False)`:
    * Initializes the environment, sets up the PyBullet connection (GUI or DIRECT), sets the time step, and initializes instance variables.
    * If `render_mode` is "human" and the GUI has not been connected, it connects to the PyBullet GUI and sets the `gui_connected` flag. Subsequent environments will connect in DIRECT mode.
* `reset(self, seed=None, options=None)`:
    * Resets the simulation to its initial state.
    * Resets the PyBullet simulation, sets gravity, loads the plane and CartPole URDF models, initializes joint velocities, and resets step counters and disturbance flags.
    * If `test_mode` is True, it randomly determines the step at which a disturbance will occur.
    * If `render_mode` is "human", it enables mouse picking in the PyBullet GUI.
    * Returns the initial observation and an empty info dictionary, consistent with the Gymnasium API.
* `disturb_pole(self, method="torque", torque_magnitude=5.0, angle_range=0.1)`:
    * Applies a disturbance to the pole.
    * Parameters:
        * `method` (str): The disturbance method, either "torque" or "tilt".
        * `torque_magnitude` (float): The magnitude of the torque applied (if `method` is "torque").
        * `angle_range` (float): The range of the random angle (if `method` is "tilt")
    * If `method` is "torque", applies a random torque to the pole joint.
    * If `method` is "tilt", resets the pole joint angle to a random value.
    * Raises a `ValueError` if an invalid disturbance method is provided.
* `_get_obs(self)`:
    * Retrieves the current observation of the environment.
    * Uses `p.getJointState()` to get the position and velocity of the cart and pole.
    * Returns a NumPy array containing the pole angle, pole velocity, cart position, and cart velocity.
* `_get_reward(self, obs)`:
    * Calculates the reward based on the pole angle.
    * The reward is 1.0 when the pole is perfectly upright (angle = 0) and decreases as the pole angle deviates from the vertical.
* `_is_done(self, obs)`:
    * Determines whether the episode has terminated.
    * The episode ends if the absolute value of the pole angle is greater than 0.5 radians or the absolute value of the cart position is greater than 2.4 meters.
* `close(self)`:
    * Disconnects from the PyBullet physics server.

#### 3.2 `CartPoleDiscreteEnv`

The `CartPoleDiscreteEnv` class inherits from `CartPoleBaseEnv` and implements the CartPole environment with a discrete action space.

##### 3.2.1 Attributes

* Inherits all attributes from `CartPoleBaseEnv`.
* `action_space`: A `gymnasium.spaces.Discrete(2)` object, representing the two discrete actions: 0 (left) and 1 (right).
* `observation_space`: A `gymnasium.spaces.Box` object defining the bounds of the observation space (pole angle, pole velocity, cart position, cart velocity).

##### 3.2.2 Methods

* `__init__(self, render_mode=False, test_mode=False)`:
    * Initializes the environment by calling the `__init__` method of the base class and setting the `action_space` and `observation_space`.
* `step(self, action)`:
    * Takes a discrete action (0 or 1) as input.
    * Applies a torque of -10.0 Nm to the cart if the action is 0, and 10.0 Nm if the action is 1. Zero torque is applied to the pole.
    * Steps the simulation using `p.stepSimulation()`.
    * Applies a disturbance using `self.disturb_pole()` if in test mode and at the disturbance step.
    * Gets the new observation, reward, and termination status using the methods defined in `CartPoleBaseEnv`.
    * Sets the truncated flag to False.
    * Records post-disturbance metrics if a disturbance has occurred.
    * Returns the observation, reward, terminated flag, truncated flag, and an empty info dictionary.

#### 3.3 `CartPoleContinuousEnv`

The `CartPoleContinuousEnv` class inherits from `CartPoleBaseEnv` and implements the CartPole environment with a continuous action space.

##### 3.3.1 Attributes

* Inherits all attributes from `CartPoleBaseEnv`.
* `action_space`: A `gymnasium.spaces.Box` object defining the continuous action space. The agent can apply torques to both the cart and the pole. The bounds are [-5.0, -5.0] and [5.0, 5.0] for the cart and pole torques, respectively.
* `observation_space`: A `gymnasium.spaces.Box` object defining the bounds of the observation space.

##### 3.3.2 Methods

* `__init__(self, render_mode=False, test_mode=False)`:
    * Initializes the environment by calling the `__init__` method of the base class and setting the `action_space` and `observation_space`.
* `step(self, action)`:
    * Takes a continuous action (a 2D NumPy array) as input, representing the torques applied to the cart and pole.
    * Applies the specified torques to the cart and pole joints using `p.setJointMotorControl2()`.
    * Steps the simulation.
    * Applies a disturbance if in test mode.
    * Gets the observation, reward, and termination status.
    * Sets `truncated` to False.
    * Records post-disturbance metrics.
    * Returns the observation, reward, terminated flag, truncated flag, and an empty info dictionary.

