import gymnasium
import numpy as np
import os
import pickle
import logging
import wandb
from datetime import datetime
from train.utils.discretizer import create_bins, discretize
from train.base_trainer import BaseTrainer

logger = logging.getLogger(__name__)

class QLearningTrainer(BaseTrainer):
    def __init__(self, config):
        self.config = config
        self.env = gymnasium.make(config.env, render_mode=None)
        self.env.action_space.seed(config.seed)
        self.env.observation_space.seed(config.seed)
        np.random.seed(config.seed)

        self.obs_low = self.env.observation_space.low
        self.obs_high = self.env.observation_space.high
        self.bins = create_bins(self.obs_low, self.obs_high, config.bins)
        self.q_table = np.zeros([config.bins] * len(self.obs_low) + [self.env.action_space.n])

        self.epsilon = config.epsilon
        self.rewards = []

        self.run_id = f"{config.algo}_{config.env}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.save_dir = os.path.join("logs", self.run_id)
        os.makedirs(self.save_dir, exist_ok=True)

        if config.resume_from and os.path.exists(config.resume_from):
            self.load(config.resume_from)

    def train(self):
        for episode in range(self.config.total_episodes):
            obs, _ = self.env.reset()
            state = discretize(obs, self.bins, self.obs_low, self.obs_high)
            total_reward = 0
            done = False

            alpha = self.config.learning_rate * (0.99 ** episode) if self.config.alpha_schedule == "decay" else self.config.learning_rate

            for _ in range(self.config.max_steps):
                if np.random.random() < self.epsilon:
                    action = self.env.action_space.sample()
                else:
                    action = np.argmax(self.q_table[state])

                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done = terminated or truncated
                next_state = discretize(next_obs, self.bins, self.obs_low, self.obs_high)

                shaped_reward = reward - 0.01 * abs(next_obs[2]) - 0.005 * abs(next_obs[3])
                best_next_action = np.max(self.q_table[next_state])

                self.q_table[state][action] += alpha * (
                    shaped_reward + self.config.discount_factor * best_next_action - self.q_table[state][action]
                )

                state = next_state
                total_reward += reward
                if done:
                    break

            self.epsilon = max(self.config.epsilon_min, self.epsilon * self.config.epsilon_decay)
            self.rewards.append(total_reward)
            # Log optimal actions and Q-value distribution
            self.optimal_actions = np.argmax(self.q_table, axis=-1)
            self.q_values_flat = self.q_table.flatten()

            wandb.log({
                "episode": episode,
                "reward": total_reward,
                "epsilon": self.epsilon,
                "alpha": alpha,
                "optimal_actions": wandb.Histogram(self.optimal_actions),
                "Q-table_max": np.max(self.q_table),
                "Q-table_min": np.min(self.q_table),
                "Q-table_mean": np.mean(self.q_table)
            })
            if episode >= 100:
                wandb.log({"moving_avg_reward": np.mean(self.rewards[-100:])})
            if episode % 1000 == 0:
                wandb.log({"Q-table": wandb.Histogram(self.q_table)})
                self.save(episode)

        self.save("final")
        self.env.close()

    def save(self, suffix):
        with open(os.path.join(self.save_dir, f"q_table_ep{suffix}.pkl"), "wb") as f:
            pickle.dump(self.q_table, f)
        logger.info(f"✅ Q-table saved at episode {suffix}")

        if suffix == "final":
            metadata = {
                "total_episodes": self.config.total_episodes,
                "epsilon_final": self.epsilon,
                "bins": self.config.bins,
                "final_reward_mean_100": np.mean(self.rewards[-100:]),
            }
            with open(os.path.join(self.save_dir, "training_metadata.pkl"), "wb") as f:
                pickle.dump(metadata, f)
            logger.info("✅ Final metadata saved.")

    def load(self, path):
        with open(path, "rb") as f:
            self.q_table = pickle.load(f)
        logger.info(f"✅ Resumed Q-table from {path}")
