import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import PPO
from gymnasium.envs.registration import register

# Append root to sys.path for module imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

# Custom imports
from train.base_trainer import BaseTrainer
from train.utils.callbacks import CustomCallback
from train.utils.logger import setup_logger
from train.utils.config_loader import load_config
import environments.cartpole  # Register custom env module
from train.utils.config_loader import get_args

logger = setup_logger()

# # Parse CLI arguments
# parser = argparse.ArgumentParser()
# parser.add_argument('--config', type=str, required=True, help='Path to config file')
# args = parser.parse_args()

# # Load config
# config = load_config(args.config)

config = get_args()

# Register custom Gymnasium environment
register(
    id="WalkerBalanceContinuousEnv",
    entry_point="environments.walker:WalkerBalanceContinuousEnv",
    max_episode_steps=500,
)

# Action noise wrapper
class ActionNoiseWrapper(gym.ActionWrapper):
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
        self.logger.info(f"[Noise] Level: {self.current_noise:.4f}, Action: {action}, Noise: {noise}")
        return clipped_action

    def step(self, action):
        self.current_noise = max(self.current_noise * self.decay_rate, self.min_noise)
        return self.env.step(action)


def main():
    # Initialize trainer
    trainer = BaseTrainer(
        algo_name="PPO",
        config=config,
        env_id="WalkerBalanceContinuousEnv",
        run_name=None,
    )

    # Wrap with action noise
    trainer.env = ActionNoiseWrapper(trainer.env)

    # Initialize model
    model = PPO(
        policy=trainer.wandb_config.policy,
        env=trainer.env,
        verbose=1,
        learning_rate=float(trainer.wandb_config.learning_rate),
        gamma=trainer.wandb_config.gamma,
        batch_size=trainer.wandb_config.batch_size,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/"
    )

    # Train
    logger.info("🚀 Starting PPO training with noise-injected environment...")
    model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_final"
    model.save(model_path)
    logger.info(f"✅ Model saved at {model_path}")

    # Log convergence episode if tracked
    if trainer.custom_callback.convergence_episode is not None:
        import wandb
        wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
        logger.info(f"📈 Logged Convergence Episode: {trainer.custom_callback.convergence_episode}")

    trainer.finish()


if __name__ == "__main__":
    main()
