## DDPG CartPole Training Script Documentation

This document provides a comprehensive overview of the `ddpg_cartpole.py` script. This script is designed to train a Deep Deterministic Policy Gradient (DDPG) agent to balance a CartPole system using a custom-defined environment within the PyBullet physics simulator. The script leverages the Stable Baselines3 library for the DDPG implementation and incorporates custom utilities for experiment management, logging, and environment setup.

### 1. Purpose

The primary purpose of this script is to:

* Set up a reinforcement learning experiment to train a DDPG agent.
* Utilize a custom CartPole environment implemented with PyBullet.
* Manage experiment configurations through external files.
* Log training progress and metrics.
* Save the trained DDPG model.

### 2. Script Description

The script performs the following key actions:

1.  **Imports Libraries:** Imports necessary libraries, including Stable Baselines3 (for the DDPG algorithm), Gymnasium (for environment management), PyBullet (through a custom environment), `argparse` (for command-line arguments), and custom modules for training utilities.

2.  **Sets up Logging:** Initializes a logger using the `setup_logger()` function from the `train.utils.logger` module.

3.  **Parses Command-Line Arguments:** Expects a command-line argument `--config` that specifies the path to a YAML configuration file.

4.  **Loads Configuration:** Loads the experiment configuration (e.g., hyperparameters) from the specified YAML file using the `load_config()` function.

5.  **Registers Custom Environment:** Registers the custom CartPole environment (`CartPoleContinuousEnv`) with Gymnasium using the `register()` function. This makes the environment accessible to Stable Baselines3. The environment is assumed to be defined in `environments.cartpole`.

6.  **Defines `main()` Function:**

    * Creates an instance of `BaseTrainer` to manage the training process. This trainer likely handles experiment setup, logging, and other common tasks.
    * Initializes a DDPG model from Stable Baselines3, passing in hyperparameters from the loaded configuration.
    * Creates an instance of a custom callback (`CustomCallback`) for monitoring training, specifically convergence.
    * Trains the DDPG agent using the `model.learn()` method.
    * Saves the trained model to a file.
    * Logs the episode at which convergence was achieved (if the training converged) to a Weights & Biases (WandB) experiment tracking platform.
    * Calls the `trainer.finish()` method for any necessary cleanup.

7.  **Executes `main()`:** The `if __name__ == "__main__":` block ensures that the `main()` function is executed when the script is run.

### 3. Modules and Dependencies

The script relies on the following modules and libraries:

* **Python Standard Library:**
    * `os`: For interacting with the operating system (e.g., file paths).
    * `sys`: For interacting with the Python interpreter (e.g., modifying the module search path).
    * `argparse`: For parsing command-line arguments.
* **Third-Party Libraries:**
    * `stable_baselines3`: Provides the DDPG algorithm.
    * `gymnasium`: Provides the environment interface.
    * `pybullet`: (Implicitly used by the custom CartPole environment) Provides the physics simulation.
* **Custom Modules (within the project):**
    * `train.utils.callbacks.CustomCallback`: A custom callback class for monitoring training progress (e.g., convergence).
    * `train.base_trainer.BaseTrainer`: A base trainer class that likely encapsulates common training loop functionality.
    * `train.utils.logger.setup_logger()`: A function to configure the logging system.
    * `train.utils.config_loader.load_config()`: A function to load experiment configurations from a file (e.g., YAML).
    * `environments.cartpole.CartPoleContinuousEnv`: A custom Gymnasium environment that interfaces with a CartPole simulation in PyBullet.

### 4. Configuration

The script uses a configuration file to manage experiment settings. The path to this file is provided as a command-line argument (`--config`). The configuration file is expected to be in YAML format and contain settings such as:

* DDPG hyperparameters (e.g., learning rate, gamma, tau, batch size, buffer size).
* Policy network architecture.
* Action noise parameters.
* Training parameters (e.g., total timesteps).
* Environment settings.

### 5. Running the Script

To run the script, execute the following command in a terminal:

    python ddpg_cartpole.py --config <path/to/your/config.yaml>

Replace `<path/to/your/config.yaml>` with the actual path to your configuration file.

### 6. Output

The script produces the following outputs:

* **Logging Information:** The script logs various information during the training process, including start and end of training, and potentially other metrics.
* **Trained Model:** The trained DDPG model is saved to a file in the `./models/ddpg/` directory. The filename includes the algorithm name.
* **TensorBoard Logs:** TensorBoard logs are saved to the `./ddpg_tensorboard/` directory, which can be used to visualize training progress in TensorBoard.
* **Convergence Episode (Optional):** If the training converges according to the `CustomCallback`, the episode number at which convergence is reached is logged to the WandB experiment tracking platform.

### 7. Customization

The script can be customized by:

* **Modifying the Configuration File:** Change the hyperparameters, network architecture, and environment settings in the YAML configuration file.
* **Modifying the Custom Modules:** Adjust the `CustomCallback`, `BaseTrainer`, and CartPole environment in the `train.utils.callbacks.py`, `train.base_trainer.py`, and `environments.cartpole.py` files, respectively, to suit specific needs.
* **Changing the Algorithm:** The script can be adapted to use other reinforcement learning algorithms from Stable Baselines3 by changing the algorithm class (e.g., `DDPG`) and adjusting the hyperparameters accordingly.
