from pendulum_env import InvertedPendulumEnv
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.callbacks import EvalCallback
import os

# ✅ Create logging directory
log_dir = "./ppo_logs/"
os.makedirs(log_dir, exist_ok=True)

# ✅ Create training environment
env = InvertedPendulumEnv(render=False)
check_env(env, warn=True)

# ✅ Create evaluation environment
eval_env = InvertedPendulumEnv(render=False)

# ✅ PPO Model
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=log_dir)

# ✅ Evaluation callback for saving best model
eval_callback = EvalCallback(
    eval_env,
    best_model_save_path="./ppo_best_model/",
    log_path=log_dir,
    eval_freq=5000,
    deterministic=True,
    render=False
)

# ✅ Train
model.learn(total_timesteps=100_000, callback=eval_callback)

# ✅ Save final model
model.save("ppo_inverted_pendulum_final")

env.close()
