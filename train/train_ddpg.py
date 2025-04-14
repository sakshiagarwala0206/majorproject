import gymnasium
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import numpy as np
import logging

# 📝 Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger()

# ✅ Register the custom environment
from gymnasium.envs.registration import register

register(
    id='AssistiveWalker-v0',
    entry_point='pendulum_env1:AssistiveWalkerEnv1',  # Change to the correct file:class path
)

# 📦 Custom callback to log training progress
class MetricsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(MetricsCallback, self).__init__(verbose)

    def _on_step(self):
        if self.n_calls % 100 == 0:
            logger.info(f"Training Step: {self.n_calls}")
        return True

# 🚀 Initialize and wrap the environment
env = gymnasium.make("AssistiveWalker-v0", render=False)
env = Monitor(env)

# 🔁 Add action noise for exploration
n_actions = env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

# 🧠 Create the DDPG model
model = DDPG("MlpPolicy", env, action_noise=action_noise, verbose=1)

# 📈 Callback to track metrics
metrics_callback = MetricsCallback()

# 🏁 Train the model
logger.info("🚀 Starting DDPG training for AssistiveWalker...")
model.learn(total_timesteps=100_000, callback=metrics_callback)

# 💾 Save the model
model.save("ddpg_assistive_walker")
logger.info("✅ Training complete. Model saved to 'ddpg_assistive_walker'")
