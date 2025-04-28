import os
import sys
from stable_baselines3 import DDPG



sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from train.utils.callbacks import CustomCallback
from train.base_trainer import BaseTrainer
from train.utils.logger import setup_logger
import argparse
from train.utils.config_loader import load_config
import gymnasium as gym
from gymnasium.envs.registration import register
import environments.cartpole
logger = setup_logger()
# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load the config
config = load_config(args.config)

# Print to verify
print(config)

register(
    id="CartPole-v1",
    entry_point="environments.cartpole:CartPoleContinuousEnv",
    max_episode_steps=500,  # same as Gym CartPole
)

def main():
    trainer = BaseTrainer(
        algo_name="DDPG",
        config=config,
        env_id="CartPole-v1",
    )

    model = DDPG(
        policy=trainer.wandb_config.policy,
        env=trainer.env,
        verbose=1,
        action_noise=trainer.action_noise,
        learning_rate=float(trainer.wandb_config.learning_rate),
        gamma=trainer.wandb_config.gamma,
        tau=trainer.wandb_config.tau,
        batch_size=trainer.wandb_config.batch_size,
        buffer_size=trainer.wandb_config.buffer_size,
        train_freq=trainer.wandb_config.train_freq,
        gradient_steps=trainer.wandb_config.gradient_steps,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
    )
    custom_callback = CustomCallback(convergence_threshold=0.5, window_size=20)
    logger.info("🚀 Starting DDPG training...")
    model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

    model.save(f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_cartpole_final")
    logger.info("✅ DDPG model saved.")
    
    if trainer.custom_callback.convergence_episode is not None:
        import wandb
        wandb.log({"Convergence Episode": trainer.custom_callback.convergence_episode})
        logger.info(f"📈 Convergence Episode logged to WandB: {trainer.custom_callback.convergence_episode}")

    trainer.finish()

if __name__ == "__main__":
    main()
