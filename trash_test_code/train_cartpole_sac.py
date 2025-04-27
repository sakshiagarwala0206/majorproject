import gymnasium
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.logger import configure
import numpy as np
import logging
import wandb
import sys
import os
from wandb.integration.sb3 import WandbCallback
from gymnasium.envs.registration import register

# Add project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# Import custom environment
from environments.cartpole_env import CartPoleEnv

# 📝 Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# ✅ Register custom environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole_env:CartPoleEnv',
)

# ✅ WandB init
wandb.init(
    project="assistive-walker-drl",
    config={
        "policy": "MlpPolicy",
        "algo": "SAC",
        "total_timesteps": 100_000,
        "learning_rate": 3e-4,
        "gamma": 0.99,
        "tau": 0.005,
        "batch_size": 256,
        "ent_coef": "auto",
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# 🎮 Environment + Monitor
env = gymnasium.make("CartPole-v1", render=False)
env = Monitor(env)

# 🧠 SAC Model
model = SAC(
    "MlpPolicy",
    env,
    verbose=1,
    tensorboard_log="./sac_tensorboard/",
    learning_rate=wandb.config.learning_rate,
    gamma=wandb.config.gamma,
    tau=wandb.config.tau,
    batch_size=wandb.config.batch_size,
    ent_coef=wandb.config.ent_coef
)

# 💾 Checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="sac_cartpole",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# 📊 WandB + Checkpoints
callback = [
    checkpoint_callback,
    WandbCallback(
        gradient_save_freq=100,
        model_save_path="./wandb_models/",
        verbose=2,
        log="all"
    )
]

from stable_baselines3.common.callbacks import BaseCallback

class TerminalMetricsCallback(BaseCallback):
    def __init__(self, verbose=1):
        super(TerminalMetricsCallback, self).__init__(verbose)

    def _on_step(self) -> bool:
        # Only log at end of episode
        if self.locals.get("dones") is not None and self.locals["dones"][0]:
            ep_info = self.locals.get("infos")[0].get("episode")
            if ep_info:
                reward = ep_info["r"]
                length = ep_info["l"]
                print(f"[EP {self.num_timesteps}] 🎯 Reward: {reward:.2f} | ⏱️ Length: {length}")
        return True
# 📊 Combine all callbacks
callback = [
    checkpoint_callback,
    WandbCallback(
        gradient_save_freq=100,
        model_save_path="./wandb_models/",
        verbose=2,
        log="all"
    ),
    TerminalMetricsCallback()
]


# 🚀 Train the SAC model
logger.info("🚀 Starting SAC training...")
model.learn(total_timesteps=wandb.config.total_timesteps, callback=callback)

# ✅ Save final model
model.save("sac_cartpole")
logger.info("✅ SAC model saved.")
wandb.finish()
