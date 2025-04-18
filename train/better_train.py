import gymnasium
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import numpy as np
import logging
import wandb
from wandb.integration.sb3 import WandbCallback
from gymnasium.envs.registration import register

# 📝 Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# ✅ Register your environment
register(
    id='AssistiveWalker-v0',
    entry_point='pendulum_env1:AssistiveWalkerEnv1',
)

# ✅ WandB init
wandb.init(
    project="assistive-walker-drl",
    config={
        "policy": "MlpPolicy",
        "algo": "DDPG",
        "total_timesteps": 100_000,
        "action_noise": 0.1,
        "gamma": 0.99,
        "tau": 0.005,
        "learning_rate": 1e-3
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# 🎮 Environment + Monitor
env = gymnasium.make("AssistiveWalker-v0", render_mode=None)
env = Monitor(env)

# 🔁 Action noise
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 🧠 Model
model = DDPG(
    "MlpPolicy",
    env,
    verbose=1,
    action_noise=action_noise,
    tensorboard_log="./ddpg_tensorboard/",
)

# 💾 Checkpoints
checkpoint_callback = CheckpointCallback(
    save_freq=10_000,
    save_path="./checkpoints/",
    name_prefix="ddpg_walker",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# 📊 WandB + Checkpoints
callback = [checkpoint_callback, WandbCallback(gradient_save_freq=100, model_save_path="./wandb_models/", verbose=2)]

# 🚀 Train
logger.info("🚀 Starting training...")
model.learn(total_timesteps=100_000, callback=callback)

# ✅ Save final model
model.save("ddpg_assistive_walker")
logger.info("✅ Model saved.")
wandb.finish()
