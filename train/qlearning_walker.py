import os
import sys
import argparse
import numpy as np
import gymnasium as gym
from datetime import datetime
import wandb
from gymnasium.envs.registration import register
from collections import deque

# Project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from train.utils.logger import setup_logger
from train.utils.config_loader import load_config
from train.base_trainer import BaseTrainer
from train.utils.qtable import convert_q_table_to_dict
import pickle

register(
    id="AssistiveWalkerDiscreteEnv-v0",
    entry_point="environments.walker:AssistiveWalkerDiscreteEnv",
    max_episode_steps=10000,
)

logger = setup_logger()
parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, required=True, help='Path to config file')
args = parser.parse_args()

# Load config
config = load_config(args.config)

# 🟡 Init wandb
wandb.init(
    project=config.get("wandb_project", "AS-Walker-Train"),
    config=config,
    mode="online"
)

class QLearningAgent:
    def __init__(self, state_space, action_space, config):
        self.state_space = state_space
        self.action_space = action_space
        self.alpha = float(config["learning_rate"])
        self.gamma = float(config["gamma"])
        self.epsilon = float(config["epsilon"])
        self.epsilon_decay = float(config["epsilon_decay"])
        self.min_epsilon = float(config["min_epsilon"])

        self.bins = config["bins"]
        self.obs_low = np.array(config["obs_low"], dtype=np.float32)
        self.obs_high = np.array(config["obs_high"], dtype=np.float32)
        self.q_table = np.zeros(self.bins + [action_space])

    def discretize_state(self, state):
        state = np.array(state, dtype=np.float32)
        ratios = (state - self.obs_low) / (self.obs_high - self.obs_low)
        ratios = np.clip(ratios, 0, 0.9999)
        indices = (ratios * self.bins).astype(int)
        return tuple(indices)

    def choose_action(self, state):
        discrete_state = self.discretize_state(state)
        if np.random.rand() < self.epsilon:
            return np.random.choice(self.action_space)
        else:
            return np.argmax(self.q_table[discrete_state])

    def update_q_value(self, state, action, reward, next_state):
        discrete_state = self.discretize_state(state)
        discrete_next_state = self.discretize_state(next_state)
        best_next_action = np.argmax(self.q_table[discrete_next_state])
        td_target = reward + self.gamma * self.q_table[discrete_next_state][best_next_action]
        td_error = td_target - self.q_table[discrete_state][action]
        self.q_table[discrete_state][action] += self.alpha * td_error

def main():

    

    trainer = BaseTrainer(
        algo_name="Q-learning",
        config=config,
        env_id="AssistiveWalkerDiscreteEnv-v0",
        run_name=None,
    )

    env = gym.make(trainer.env_id)
    from gymnasium import spaces
    import random
    seed = config.get("seed")
    np.random.seed(seed)
    random.seed(seed)
    env.reset(seed=seed)
    if isinstance(env.observation_space, spaces.Discrete):
        state_space = env.observation_space.n
    elif isinstance(env.observation_space, spaces.Box):
        state_space = env.observation_space.shape[0]
    else:
        raise ValueError("Unsupported observation space type.")

    agent = QLearningAgent(state_space=state_space, action_space=env.action_space.n, config=config)

    total_episodes = int(config["total_episodes"])
    max_steps = int(config["max_steps_per_episode"])
    recent_rewards = deque(maxlen=100)
    convergence_logged = False

    for episode in range(total_episodes):
        full_obs, info = env.reset()
        state = np.array(full_obs[:7], dtype=np.float32)  # Only use first 7 for agent
        total_rewards = 0
        start_time = datetime.now()

        for step in range(max_steps):
            action = agent.choose_action(state)
            full_next_obs, reward, done, truncated, info = env.step(action)
            next_state = np.array(full_next_obs[:7], dtype=np.float32)  # Only use first 7 for agent
            agent.update_q_value(state, action, reward, next_state)
            state = next_state
            total_rewards += reward

            # Log IMU data (elements 0-8 of info["imu"], corresponding to obs[7:16])
            if "imu" in info:
                imu = info["imu"]
                wandb.log({
                    "imu_roll": imu[0],
                    "imu_pitch": imu[1],
                    "imu_yaw": imu[2],
                    "imu_ang_vel_x": imu[3],
                    "imu_ang_vel_y": imu[4],
                    "imu_ang_vel_z": imu[5],
                    "imu_lin_acc_x": imu[6],
                    "imu_lin_acc_y": imu[7],
                    "imu_lin_acc_z": imu[8],
                    "step": step,
                    "episode": episode + 1
                })

            if done or truncated:
                break

        agent.epsilon = max(agent.epsilon * agent.epsilon_decay, agent.min_epsilon)
        episode_length = step + 1
        recent_rewards.append(total_rewards)
        avg_reward_100 = np.mean(recent_rewards) if len(recent_rewards) == 100 else None

        # 🟢 wandb logging
        wandb.log({
            "episode": episode + 1,
            "total_reward": total_rewards,
            "epsilon": agent.epsilon,
            "episode_length": episode_length,
            "average_reward_100": avg_reward_100,
            "max_q_value": np.max(agent.q_table),
            "mean_q_value": np.mean(agent.q_table),
            "q_table_std": np.std(agent.q_table),
            "time_per_episode": (datetime.now() - start_time).total_seconds(),
        })

        # 🎯 Convergence logging
        if not convergence_logged and avg_reward_100 is not None and "solved_reward" in config and avg_reward_100 >= config["solved_reward"]:
            wandb.log({"convergence_episode": episode + 1})
            logger.info(f"📈 Environment solved at episode {episode + 1}")
            convergence_logged = True

        logger.info(f"Episode {episode+1}/{total_episodes}, Reward: {total_rewards:.2f}, Epsilon: {agent.epsilon:.4f}")

    # 💾 Save Q-table
    q_table_dict = convert_q_table_to_dict(agent.q_table, agent.action_space)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(f"./final_model/{trainer.algo_name.lower()}/", exist_ok=True)
    model_path = f"./final_model/{trainer.algo_name.lower()}/{trainer.algo_name.lower()}_{timestamp}_agent.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(q_table_dict, f)
    logger.info(f"✅ Q-learning agent saved at {model_path}")

    trainer.finish()

if __name__ == "__main__":
    main()


