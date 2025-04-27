import os
import sys
from stable_baselines3 import DQN

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from train.base_trainer import BaseTrainer
from train.utils.logger import setup_logger

logger = setup_logger()
import argparse
from train.utils.config_loader import load_config

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load the config
config = load_config(args.config)

# Print to verify
print(config)

# Now you can pass config to your trainer class
# trainer = DQNTrainer(config)
# trainer.train()
controller_name = "DQN"
config_name = "EpsDecay_0.99"




def main():
    trainer = BaseTrainer(
        algo_name="DQN",
        config=config,
        env_id="CartPole-v1",
    )

    model = DQN(
        policy=trainer.wandb_config.policy,
        env=trainer.env,
        verbose=1,
        learning_rate=float(trainer.wandb_config.learning_rate),
        gamma=trainer.wandb_config.gamma,
        batch_size=trainer.wandb_config.batch_size,
        buffer_size=trainer.wandb_config.buffer_size,
        learning_starts=trainer.wandb_config.learning_starts,
        train_freq=trainer.wandb_config.train_freq,
        target_update_interval=trainer.wandb_config.target_update_interval,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
    )

    logger.info("🚀 Starting DQN training...")
    model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

    model.save(f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_cartpole_final")
    logger.info("✅ DQN model saved.")

    trainer.finish()

if __name__ == "__main__":
    main()
