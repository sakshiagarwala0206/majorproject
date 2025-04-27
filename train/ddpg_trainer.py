import os
import sys
from stable_baselines3 import DDPG

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

from train.base_trainer import BaseTrainer
from train.utils.logger import setup_logger

logger = setup_logger()

def main():
    trainer = BaseTrainer(
        algo_name="DDPG",
        config_path=os.path.join("configs", "ddpg.yaml"),
        env_id="CartPole-v1",
    )

    model = DDPG(
        policy=trainer.wandb_config.policy,
        env=trainer.env,
        verbose=1,
        action_noise=trainer.action_noise,
        learning_rate=trainer.wandb_config.learning_rate,
        gamma=trainer.wandb_config.gamma,
        tau=trainer.wandb_config.tau,
        batch_size=trainer.wandb_config.batch_size,
        buffer_size=trainer.wandb_config.buffer_size,
        train_freq=trainer.wandb_config.train_freq,
        gradient_steps=trainer.wandb_config.gradient_steps,
        tensorboard_log=f"./{trainer.algo_name.lower()}_tensorboard/",
    )

    logger.info("🚀 Starting DDPG training...")
    model.learn(total_timesteps=trainer.wandb_config.total_timesteps, callback=trainer.get_callbacks())

    model.save(f"./models/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_cartpole_final")
    logger.info("✅ DDPG model saved.")
    
    trainer.finish()

if __name__ == "__main__":
    main()
