import gymnasium
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor
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

# Now you can import the environment
from environments.cartpole_env import CartPoleEnv  # Import the CartPoleEnv class

# 📝 Logging config
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# ✅ Register your environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole_env:CartPoleEnv',  # Correct the entry point
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
        "learning_rate": 1e-3,
    },
    sync_tensorboard=True,
    monitor_gym=True,
    save_code=True,
)

# 🎮 Environment + Monitor
env = gymnasium.make("CartPole-v1", render=False)  # Use your custom environment
# env = CartPoleEnv(render=False)  # Use your custom environment
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
    name_prefix="ddpg_cartpole",
    save_replay_buffer=True,
    save_vecnormalize=True,
)

# 📊 WandB + Checkpoints
callback = [checkpoint_callback, WandbCallback(gradient_save_freq=100, model_save_path="./wandb_models/", verbose=2)]

# 🚀 Train
logger.info("🚀 Starting training...")
model.learn(total_timesteps=100_000, callback=callback)

# ✅ Save final model
model.save("ddpg_cartpole")
logger.info("✅ Model saved.")
wandb.finish()
