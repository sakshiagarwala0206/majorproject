import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO
from gymnasium.envs.registration import register

# Project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from train.base_trainer import BaseTrainer
from train.utils.callbacks import CustomCallback
from train.utils.logger import setup_logger
from train.utils.config_loader import load_config
import environments.cartpole  # Ensure this import is valid


logger = setup_logger()

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Register custom environment
register(
    id="InclinedCartPole",
    entry_point="environments.CartPoleOnInclinedPlaneEnv:InclinedCartPoleContinuousEnv",
    max_episode_steps=500,
)


class ActionNoiseWrapper(gym.ActionWrapper):
    """Adds Gaussian noise to actions and logs how much noise was added."""
    def __init__(self, env, initial_noise=0.5, decay_rate=0.999, min_noise=0.05):
        super().__init__(env)
        self.initial_noise = initial_noise
        self.decay_rate = decay_rate
        self.min_noise = min_noise
        self.current_noise = initial_noise
        self.logger = setup_logger()

    def action(self, action):
        noise = np.random.normal(0, self.current_noise, size=action.shape)
        noisy_action = action + noise
        clipped_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)

        # Log the noise and the difference from the clean action
        self.logger.info(f"Noise Level: {self.current_noise:.4f}, "
                         f"Original Action: {action}, "
                         f"Noise: {noise}, "
                         f"Noisy Action: {clipped_action}")

        return clipped_action

    def step(self, action):
        self.current_noise = max(self.current_noise * self.decay_rate, self.min_noise)
        return self.env.step(action)


def main():
    trainer = BaseTrainer(
        algo_name="PPO",
        config=config,
        env_id="InclinedCartPole",
        run_name=None,
    )

    # Wrap environment with noise wrapper
    noisy_env = ActionNoiseWrapper(trainer.env)

    # Replace trainer.env with wrapped one to keep consistency
    trainer.env = noisy_env

    model = PPO(
        policy=trainer.wandb_config.policy,
        env=noisy_env,
        verbose=1,
        learning_rate=float(trainer.wandb_config.learning_rate),
        gamma=trainer.wandb_config.gamma,
        batch_size=trainer.wandb_config.batch_size,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
    )

    logger.info("🚀 Starting PPO training with action noise and noise logging...")
    model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

    # Save trained model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_cartpole_final"
    model.save(model_path)
    logger.info(f"✅ PPO model saved at {model_path}")

    # Log convergence episode if it exists
    if trainer.custom_callback.convergence_episode is not None:
        import wandb
        wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
        logger.info(f"📈 Convergence Episode logged to WandB: {trainer.custom_callback.convergence_episode}")

    trainer.finish()


if __name__ == "__main__":
    main()









# import os
# import sys
# import argparse
# import numpy as np
# import gymnasium as gym
# from datetime import datetime
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.env_util import make_vec_env

# # Project-specific imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from train.base_trainer import BaseTrainer
# from train.utils.callbacks import CustomCallback
# from train.utils.logger import setup_logger
# from train.utils.config_loader import load_config
# import environments.cartpole  # Custom env
# from gymnasium.envs.registration import register

# logger = setup_logger()

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True, help='Path to config file')
# args = parser.parse_args()

# # Load config
# config = load_config(args.config)

# # Register environment-inclined environment
# register(
#     id="InclinedCartPole",
#     entry_point="environments.CartPoleOnInclinedPlaneEnv:InclinedCartPoleContinuousEnv",
#     max_episode_steps=500,
# )
# class CartPoleLoggingWrapper(gym.Wrapper):
#     """Wrapper to log specific states (pole angle, pole velocity, cart position, and cart velocity)."""
#     def __init__(self, env):
#         super().__init__(env)
#         self.logger = setup_logger()  # Set up a logger

#     def reset(self, **kwargs):
#         state = self.env.reset(**kwargs)
#         self.log_state(state)  # Log initial state
#         return state

#     def step(self, action):
#         state, reward, done, truncated, info = self.env.step(action)
#         self.log_state(state)  # Log state after each step
#         return state, reward, done, truncated, info

#     def log_state(self, state):
#         # Assuming the state is in the order [cart_position, cart_velocity, pole_angle, pole_velocity]
#         cart_position, cart_velocity, pole_angle, pole_velocity = state
#         self.logger.info(f"State Log - Cart Position: {cart_position}, Cart Velocity: {cart_velocity}, "
#                          f"Pole Angle: {pole_angle}, Pole Velocity: {pole_velocity}")
        
# class ActionNoiseWrapper(gym.ActionWrapper):
#     """Wraps env to add noise to agent actions."""
#     def __init__(self, env, initial_noise=0.5, decay_rate=0.999, min_noise=0.05):
#         super().__init__(env)
#         self.initial_noise = initial_noise
#         self.decay_rate = decay_rate
#         self.min_noise = min_noise
#         self.current_noise = initial_noise

#     def action(self, action):
#         noise = np.random.normal(0, self.current_noise, size=action.shape)
#         noisy_action = action + noise
#         clipped_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
#         return clipped_action

#     def step(self, action):
#         self.current_noise = max(self.current_noise * self.decay_rate, self.min_noise)
#         return self.env.step(action)

# def main():
#     trainer = BaseTrainer(
#         algo_name="PPO",
#         config=config,
#         env_id="InclinedCartPole",
#         run_name=None,
#     )

#     # Wrap env to add action noise
#     noisy_env = ActionNoiseWrapper(trainer.env)
#     logged_env = CartPoleLoggingWrapper(noisy_env)  # Log states while also adding action noise
#     model = PPO(
#         policy=trainer.wandb_config.policy,
#         env=noisy_env,
#         verbose=1,
#         learning_rate=float(trainer.wandb_config.learning_rate),
#         gamma=trainer.wandb_config.gamma,
#         batch_size=trainer.wandb_config.batch_size,
#         tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
#     )

#     logger.info("🚀 Starting PPO training with action noise...")
#     model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

#     # Save model
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_cartpole_final"
#     model.save(model_path)
#     logger.info(f"✅ PPO model saved at {model_path}")

#     # Log convergence episode if exists
#     if trainer.custom_callback.convergence_episode is not None:
#         import wandb
#         wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
#         logger.info(f"📈 Convergence Episode logged to WandB: {trainer.custom_callback.convergence_episode}")

#     trainer.finish()

# if __name__ == "__main__":
#     main()



# import os
# import sys
# import argparse
# import numpy as np
# import gymnasium as gym
# from datetime import datetime
# from stable_baselines3 import PPO
# from stable_baselines3.common.callbacks import BaseCallback
# from stable_baselines3.common.env_util import make_vec_env

# # Project-specific imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from train.base_trainer import BaseTrainer
# from train.utils.callbacks import CustomCallback
# from train.utils.logger import setup_logger
# from train.utils.config_loader import load_config
# import environments.cartpole  # Custom env
# from gymnasium.envs.registration import register

# logger = setup_logger()

# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True, help='Path to config file')
# args = parser.parse_args()

# # Load config
# config = load_config(args.config)

# # # Register environment-plain environment
# # register(
# #     id="CartPole-v1",
# #     entry_point="environments.cartpole:CartPoleContinuousEnv",
# #     max_episode_steps=500,
# # )

# # Register environment-inclined environment
# register(
#     id="CartPole-v1",
#     entry_point="environments.CartPoleOnInclinedPlaneEnv:InclinedCartPoleContinuousEnv",
#     max_episode_steps=500,
# )
# class ActionNoiseWrapper(gym.ActionWrapper):
#     """Wraps env to add noise to agent actions."""
#     def __init__(self, env, initial_noise=0.5, decay_rate=0.999, min_noise=0.05):
#         super().__init__(env)
#         self.initial_noise = initial_noise
#         self.decay_rate = decay_rate
#         self.min_noise = min_noise
#         self.current_noise = initial_noise

#     def action(self, action):
#         noise = np.random.normal(0, self.current_noise, size=action.shape)
#         noisy_action = action + noise
#         clipped_action = np.clip(noisy_action, self.action_space.low, self.action_space.high)
#         return clipped_action

#     def step(self, action):
#         self.current_noise = max(self.current_noise * self.decay_rate, self.min_noise)
#         return self.env.step(action)

# def main():
#     trainer = BaseTrainer(
#         algo_name="PPO",
#         config=config,
#         env_id="CartPole-v1",
#         run_name=None,
#     )

#     # Wrap env to add action noise
#     noisy_env = ActionNoiseWrapper(trainer.env)

#     model = PPO(
#         policy=trainer.wandb_config.policy,
#         env=noisy_env,
#         verbose=1,
#         learning_rate=float(trainer.wandb_config.learning_rate),
#         gamma=trainer.wandb_config.gamma,
#         batch_size=trainer.wandb_config.batch_size,
#         tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
#     )

#     logger.info("🚀 Starting PPO training with action noise...")
#     model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

#     # Save model
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     model_path = f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_cartpole_final"
#     model.save(model_path)
#     logger.info(f"✅ PPO model saved at {model_path}")

#     # Log convergence episode if exists
#     if trainer.custom_callback.convergence_episode is not None:
#         import wandb
#         wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
#         logger.info(f"📈 Convergence Episode logged to WandB: {trainer.custom_callback.convergence_episode}")

#     trainer.finish()

# if __name__ == "__main__":
#     main()

# import os
# import sys
# from stable_baselines3 import PPO
# from stable_baselines3.common.noise import NormalActionNoise


# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
# from train.utils.callbacks import CustomCallback
# from train.base_trainer import BaseTrainer
# from train.utils.logger import setup_logger
# import argparse
# from train.utils.config_loader import load_config
# import gymnasium as gym
# from gymnasium.envs.registration import register
# import numpy as np
# import environments.cartpole
# from stable_baselines3.common.callbacks import BaseCallback
# logger = setup_logger()
# # Parse command-line arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True, help='Path to config file')
# args = parser.parse_args()

# # Load the config
# config = load_config(args.config)

# # Print to verify
# print(config)

# register(
#     id="CartPole-v1",
#     entry_point="environments.cartpole:CartPoleContinuousEnv",
#     max_episode_steps=500,  # same as Gym CartPole
# )

# # Define the initial noise parameters
# initial_noise = 1.0     # Starting noise level
# decay_rate = 0.999      # Rate at which the noise decays
# min_noise = 0.05        # Minimum noise level

# def main():
#     trainer = BaseTrainer(
#         algo_name="PPO",
#         config=config,
#         env_id="CartPole-v1",
#         run_name=None,
#     )
#     # Extract the environment's action space shape to define the action noise
#     action_space_shape = trainer.env.action_space.shape
#     # action_noise = NormalActionNoise(mean=np.zeros(action_space_shape), sigma=np.ones(action_space_shape) * initial_noise)

#     model = PPO(
#         policy=trainer.wandb_config.policy,
#         env=trainer.env,
#         verbose=1,
#         # action_noise=action_noise,
#         learning_rate=float(trainer.wandb_config.learning_rate),
#         gamma=trainer.wandb_config.gamma,
#         # tau=trainer.wandb_config.tau,
#         batch_size=trainer.wandb_config.batch_size,
#         # buffer_size=trainer.wandb_config.buffer_size,
#         # train_freq=trainer.wandb_config.train_freq,
#         # gradient_steps=trainer.wandb_config.gradient_steps,
#         tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
#     )
    
#     custom_callback = CustomCallback(convergence_threshold=0.5, window_size=20)
#     logger.info("🚀 Starting PPO training...")
#     model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

#     import datetime
#     timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     model.save(f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_cartpole_final")
#     logger.info("✅ PPO model saved.")
    
#     if trainer.custom_callback.convergence_episode is not None:
#         import wandb
#         wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
#         logger.info(f"📈 Convergence Episode logged to WandB: {trainer.custom_callback.convergence_episode}")

#     trainer.finish()

# if __name__ == "__main__":
#     main()
