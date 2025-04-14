import gymnasium
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import logging

# 📝 Logging setup
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# ✅ Register your custom environment
from gymnasium.envs.registration import register

register(
    id='InvertedPendulum-v0',
    entry_point='cartpole_env:InvertedPendulumEnvGymnasium',  # Replace 'your_env_file' with the actual .py filename (without .py)
)

# 📦 Custom training metrics callback
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)

    def _on_step(self):
        if self.n_calls % 100 == 0:
            logger.info(f"Training Step: {self.n_calls}")
        return True

# 🚀 Initialize and wrap the environment
env = gymnasium.make("InvertedPendulum-v0", render_mode=None)
env = Monitor(env)

# 🔁 Action noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 🧠 Instantiate DDPG model
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# 📈 Metrics callback instance
metrics_callback = MetricsCallback()

# 🏁 Train the model
logger.info("🚀 Starting DDPG training for Inverted Pendulum...")
model.learn(total_timesteps=100_000, callback=metrics_callback)

# 💾 Save the trained model
model.save("ddpg_inverted_pendulum")
logger.info("✅ Training complete. Model saved as 'ddpg_inverted_pendulum'")
