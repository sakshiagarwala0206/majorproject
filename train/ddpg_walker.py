import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
from stable_baselines3 import DDPG
from gymnasium.envs.registration import register

# Project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from train.base_trainer import BaseTrainer
from train.utils.callbacks import CustomCallback
from train.utils.logger import setup_logger
from train.utils.config_loader import load_config
import environments.walker
from train.base_trainer import StopTrainingOnPatience
logger = setup_logger()

import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# Register custom environment
register(
    id="AssistiveWalkerContinuousEnv-v0",
    entry_point="environments.walker_1:AssistiveWalkerContinuousEnv",
    max_episode_steps=10000,
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
        algo_name="DDPG",
        config=config,
        env_id="AssistiveWalkerContinuousEnv-v0",
        run_name=None,
    )

    # Initialize wandb for online logging
    wandb.init(
        project=trainer.wandb_config.wandb_project,
        entity=getattr(trainer.wandb_config, "wandb_entity", None),
        config=vars(trainer.wandb_config),
        sync_tensorboard=True,
        mode="online"
    )

    # Wrap environment with noise wrapper
    noisy_env = ActionNoiseWrapper(trainer.env)
    trainer.env = noisy_env

    # Initialize DDPG with parameters from config
    model = DDPG(
        policy=trainer.wandb_config.policy,
        env=noisy_env,
        verbose=1,
        learning_rate=float(trainer.wandb_config.learning_rate),
        buffer_size=trainer.wandb_config.buffer_size,
        learning_starts=trainer.wandb_config.learning_starts,
        batch_size=trainer.wandb_config.batch_size,
        tau=trainer.wandb_config.tau,
        gamma=trainer.wandb_config.gamma,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
    )

    logger.info("🚀 Starting DDPG training with action noise and noise logging...")
    patience_callback = StopTrainingOnPatience(min_improvement=1.0, patience=200000)
    callbacks = [patience_callback] + list(trainer.get_callbacks())
    model.learn(
        total_timesteps=trainer.wandb_config.total_timesteps,
        callback=callbacks,
        log_interval=4
    )

    # Save only the final trained model (policy and weights, no replay buffer or optimizer)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_assistivewalker_final"
    model.save(model_path, exclude=["replay_buffer", "optimizer"])
    logger.info(f"✅ DDPG model saved at {model_path}")

    # Log convergence episode to wandb (if available)
    if hasattr(trainer, "custom_callback") and getattr(trainer.custom_callback, "convergence_episode", None) is not None:
        wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
        logger.info(f"📈 Convergence Episode logged to WandB: {trainer.custom_callback.convergence_episode}")

    trainer.finish()
    wandb.finish()

if __name__ == "__main__":
    main()