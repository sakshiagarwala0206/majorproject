import os
import sys
import gymnasium
import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
from gymnasium.envs.registration import register

# 🔧 Add project root to import custom env
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

# ✅ Import and register your custom environment
from environments.cartpole_rl import CartPoleEnv

register(
    id='CartPole-v1',
    entry_point='environments.cartpole_rl:CartPoleEnv',
)

# ✅ Load Q-table
with open("q_table_final.pkl", "rb") as f:
    q_table = pickle.load(f)

# 🎮 Create the environment
env = gymnasium.make("CartPole-v1", render_mode="human")  # use "rgb_array" for headless
obs_low = env.observation_space.low
obs_high = env.observation_space.high

# ⚙️ Define bins (should match training)
bins = 10  # or load from saved config
def create_bins(low, high, bins):
    return [np.linspace(l, h, bins - 1) for l, h in zip(low, high)]

def discretize(obs, bins):
    return tuple(int(np.digitize(x, b)) for x, b in zip(obs, bins))

obs_bins = create_bins(obs_low, obs_high, bins)

# 📈 Evaluation loop
n_eval_episodes = 10
rewards = []

print("🎯 Starting Q-Learning Evaluation...")

for ep in range(n_eval_episodes):
    obs, _ = env.reset()
    state = discretize(obs, obs_bins)
    total_reward = 0
    done = False

    while not done:
        action = np.argmax(q_table[state])
        obs, reward, terminated, truncated, _ = env.step(action)
        state = discretize(obs, obs_bins)
        total_reward += reward
        done = terminated or truncated

        # Delay for visualization
        time.sleep(0.01)

    rewards.append(total_reward)
    print(f"✅ Episode {ep + 1}: Total Reward = {total_reward}")

env.close()

# 📊 Evaluation summary
mean_reward = np.mean(rewards)
max_reward = np.max(rewards)
min_reward = np.min(rewards)

print("\n📊 Q-Learning Evaluation Summary:")
print(f"Average Reward: {mean_reward:.2f}")
print(f"Max Reward: {max_reward}")
print(f"Min Reward: {min_reward}")

# 📉 Plot reward trend
plt.plot(range(n_eval_episodes), rewards, marker='o', label="Total Reward")
plt.title("Q-Learning Evaluation Rewards per Episode")
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
