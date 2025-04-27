import os
import sys
from stable_baselines3 import PPO

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from train.base_trainer import BaseTrainer
from train.utils.logger import setup_logger

logger = setup_logger()

def main():
    trainer = BaseTrainer(
        algo_name="PPO",
        config_path=os.path.join("configs", "ppo.yaml"),
        env_id="CartPole-v1",
    )

    model = PPO(
        policy=trainer.wandb_config.policy,
        env=trainer.env,
        verbose=1,
        learning_rate=trainer.wandb_config.learning_rate,
        gamma=trainer.wandb_config.gamma,
        batch_size=trainer.wandb_config.batch_size,
        n_steps=trainer.wandb_config.n_steps,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
    )

    logger.info("🚀 Starting PPO training...")
    model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

    model.save(f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_cartpole_final")
    logger.info("✅ PPO model saved.")

    trainer.finish()

if __name__ == "__main__":
    main()
