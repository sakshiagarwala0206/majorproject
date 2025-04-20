import gymnasium
import numpy as np
import os
import sys
import pickle
import argparse
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb

# 📁 Add root for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole_rl import CartPoleEnv  # Custom env
from train.utils.discretizer import create_bins, discretize
from train.utils.qtable_utils import save_q_table, load_q_table, save_metadata
from train.utils.logger import setup_logger

# 📝 Logger
logger = setup_logger()

# ✅ Register the environment
gym_id = 'CartPole-v1'
gymnasium.envs.registration.register(
    id=gym_id,
    entry_point='environments.cartpole_rl:CartPoleEnv',
)

# 📄 Config loader
def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

# 🚀 Main Q-learning function
def train_q_learning(config_path):
    config = load_config(config_path)

    # 🎯 WandB
    wandb.init(
        project="assistive-walker-qlearning",
        config=vars(config),
        save_code=True,
    )
    config = wandb.config

    # 📁 Save directory
    run_id = f"{config.algo}_{config.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_dir = os.path.join("logs", run_id)
    os.makedirs(save_dir, exist_ok=True)

    # 🎮 Env setup
    np.random.seed(config.seed)
    env = gymnasium.make(config.env, render_mode=None)
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)

    obs_low, obs_high = env.observation_space.low, env.observation_space.high
    bins = create_bins(obs_low, obs_high, config.bins)
    q_table = np.zeros([config.bins] * len(obs_low) + [env.action_space.n])

    # 🔁 Resume
    resume_from = getattr(config, "resume_from", None)
    if resume_from and os.path.exists(resume_from):
        q_table = load_q_table(resume_from)
        logger.info(f"✅ Resumed Q-table from {resume_from}")

    epsilon = config.epsilon
    rewards = []

    # 🧠 Training loop
    for episode in range(config.total_episodes):
        obs, _ = env.reset()
        state = discretize(obs, bins, obs_low, obs_high)
        total_reward = 0

        alpha = config.learning_rate * (0.99 ** episode) if config.alpha_schedule == "decay" else config.learning_rate

        for _ in range(config.max_steps):
            action = env.action_space.sample() if np.random.rand() < epsilon else np.argmax(q_table[state])

            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = discretize(next_obs, bins, obs_low, obs_high)

            shaped_reward = reward - 0.01 * abs(next_obs[2]) - 0.005 * abs(next_obs[3])
            best_next = np.max(q_table[next_state])

            q_table[state][action] += alpha * (shaped_reward + config.discount_factor * best_next - q_table[state][action])

            state = next_state
            total_reward += reward
            if done:
                break

        epsilon = max(config.epsilon_min, epsilon * config.epsilon_decay)
        rewards.append(total_reward)

        wandb.log({
            "episode": episode,
            "reward": total_reward,
            "epsilon": epsilon,
            "alpha": alpha,
        })
        if episode >= 100:
            wandb.log({"moving_avg_reward": np.mean(rewards[-100:])})
        if episode % 1000 == 0:
            wandb.log({"Q-table": wandb.Histogram(q_table)})

        # 💾 Checkpoint
        if episode % config.checkpoint_interval == 0 and episode > 0:
            save_q_table(q_table, episode, save_dir)
            logger.info(f"💾 Q-table saved at episode {episode}")

    # ✅ Final save
    save_q_table(q_table, "final", save_dir)
    save_metadata({
        "total_episodes": config.total_episodes,
        "epsilon_final": epsilon,
        "bins": config.bins,
        "final_reward_mean_100": np.mean(rewards[-100:]),
    }, save_dir)

    logger.info("✅ Training complete.")
    env.close()
    wandb.finish()

# 🔧 Entrypoint
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    train_q_learning(args.config)
