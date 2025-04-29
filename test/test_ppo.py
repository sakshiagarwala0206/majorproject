import argparse
import gymnasium
import numpy as np
import torch
import os
import sys
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb
from gymnasium.envs.registration import register
import random

# 📁 Add root for custom modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole import CartPoleContinuousEnv
from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import PPO

logger = setup_logger()

# ✅ Register custom environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole:CartPoleContinuousEnv',
)

# 🧾 Config loader
def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

# 📈 Evaluation metrics
def get_convergence_time(rewards, threshold=0.95):
    for episode, reward in enumerate(rewards):
        if reward >= threshold:
            return episode
    return len(rewards)

def get_avg_reward(rewards):
    return float(np.mean(rewards))

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles))

def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
    for t, a in zip(times, angles):
        if abs(a - target_angle) <= tolerance:
            return float(t)
    return float(max(times))

def get_fall_rate(fall_count, total_episodes):
    return float((fall_count / total_episodes) * 100)

def get_energy(torques):
    flat = []
    for v in torques:
        try:
            vals = list(v)
        except TypeError:
            vals = [v]
        flat.extend([abs(x) for x in vals])
    return float(sum(flat))

def get_smoothness(angles, time_steps):
    jerks = np.diff(angles) / np.diff(time_steps)
    return float(np.mean(np.abs(jerks)))

def get_robustness(rewards):
    return float(np.std(rewards))

# 🧪 Evaluation loop
def evaluate_controller(env, controller, num_episodes, max_steps,
                        epsilon_start=1.0, epsilon_min=0.7, epsilon_decay=0.999):
    rewards = []
    overshoots = []
    settling_times = []
    fall_count = 0
    energy_list = []
    smoothness_list = []
    epsilon = epsilon_start

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        episode_torques = []
        episode_angles = []
        episode_times = []

        for step in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action, _ = controller.model.predict(obs, deterministic=True)
            action_arr = np.array(action, dtype=float).flatten()

            obs, reward, terminated, truncated, info = env.step(action_arr)
            done = terminated or truncated

            total_reward += reward
            episode_times.append(step)
            episode_torques.append(action_arr.tolist())
            episode_angles.append(obs[2])

            if done:
                if abs(obs[2]) > 0.8:
                    fall_count += 1
                break

        rewards.append(total_reward)
        overshoots.append(get_overshoot(episode_angles))
        settling_times.append(get_settling_time(episode_times, episode_angles))
        energy_list.append(get_energy(episode_torques))
        smoothness_list.append(get_smoothness(episode_angles, episode_times))

        epsilon = max(epsilon_min, epsilon * epsilon_decay)

    metrics = {
        "Convergence Time (ep)": get_convergence_time(rewards),
        "Avg Reward": get_avg_reward(rewards),
        "Overshoot (°)": float(np.mean(overshoots)),
        "Settling Time (steps)": float(np.mean(settling_times)),
        "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
        "Energy (∑|τ|)": float(np.mean(energy_list)),
        "Smoothness (Jerk)": float(np.mean(smoothness_list)),
        "Robustness": get_robustness(rewards)
    }

    return rewards, metrics, overshoots, settling_times, energy_list, smoothness_list

# 🚀 Main evaluation script
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
    parser.add_argument('--model_path', type=str, required=True, help="Path to PPO model (.zip)")
    parser.add_argument('--use_wandb', action='store_true', help="Log results to wandb")
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="assistive-walker-eval-DRL",
            config=vars(config),
            name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
        )

    # create env
    env = gymnasium.make("CartPole-v1", render_mode="human")
    env.action_space.seed(config.seed)
    env.observation_space.seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)

    # load model & controller
    model = PPO.load(args.model_path)
    controller = DRLController(model, "PPO", action_space=env.action_space)

    logger.info(f"🔍 Evaluating PPO controller...")

    rewards, metrics, overshoots, settling_times, energy_list, smoothness_list = \
        evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
    mean_reward = float(np.mean(rewards))

    logger.info(f"📊 Mean reward over {config.eval_episodes} episodes: {mean_reward}")

    # 🗃️ Log to WandB if enabled
    if args.use_wandb:
        # Log summary metrics
        wandb.log({"mean_reward": mean_reward, **metrics})

        # Create tables and line plots
        data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
        cols = ["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"]
        episode_table = wandb.Table(data=data, columns=cols)

        wandb.log({
            "Reward Curve": wandb.plot.line(episode_table, "episode", "reward", title="Reward per Episode"),
            "Overshoot Curve": wandb.plot.line(episode_table, "episode", "overshoot", title="Overshoot per Episode"),
            "Settling Time Curve": wandb.plot.line(episode_table, "episode", "settling_time", title="Settling Time per Episode"),
            "Energy Curve": wandb.plot.line(episode_table, "episode", "energy", title="Energy per Episode"),
            "Smoothness Curve": wandb.plot.line(episode_table, "episode", "smoothness", title="Smoothness per Episode")
        })

        # Log the raw table as well
        wandb.log({"evaluation_table": episode_table})
        wandb.finish()

    env.close()



# import argparse
# import gymnasium
# import numpy as np
# import torch
# import os
# import sys
# import yaml
# from types import SimpleNamespace
# from datetime import datetime
# import wandb
# from gymnasium.envs.registration import register
# import random

# # 📁 Add root for custom modules
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from environments.cartpole import CartPoleContinuousEnv
# from controllers.drl_controller import DRLController
# from train.utils.logger import setup_logger
# from stable_baselines3 import PPO

# logger = setup_logger()

# # ✅ Register custom environment
# register(
#     id='CartPole-v1',
#     entry_point='environments.cartpole:CartPoleContinuousEnv',
# )

# # 🧾 Config loader
# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config_dict = yaml.safe_load(f)
#     return SimpleNamespace(**config_dict)

# # 📈 Evaluation metrics
# def get_convergence_time(rewards, threshold=0.95):
#     for episode, reward in enumerate(rewards):
#         if reward >= threshold:
#             return episode
#     return len(rewards)

# def get_avg_reward(rewards):
#     return np.mean(rewards)

# def get_overshoot(angles, target_angle=0):
#     return max(abs(angle - target_angle) for angle in angles)

# def get_settling_time(times, angles, target_angle=0, tolerance=0.05):
#     for t, angle in zip(times, angles):
#         if abs(angle - target_angle) <= tolerance:
#             return t
#     return max(times)

# def get_fall_rate(falls, total_episodes):
#     return (falls / total_episodes) * 100

# def get_energy(torques):
#     # flatten and sum to a single Python float
#     flat = [abs(v) for vec in torques for v in vec]
#     energy = float(sum(flat))

# def get_smoothness(angles, time_steps):
#     jerks = np.diff(angles) / np.diff(time_steps)
#     return np.mean(np.abs(jerks))

# def get_robustness(performances):
#     return np.std(performances)

# # 🧪 Evaluation loop
# def evaluate_controller(env, controller, num_episodes, max_steps, epsilon_start=1.0, epsilon_min=0.7, epsilon_decay=0.999):
#     rewards = []
#     overshoots = []
#     settling_times = []
#     fall_count = 0
#     energy = []
#     smoothness = []
#     epsilon = epsilon_start

#     for episode in range(num_episodes):
#         obs, _ = env.reset(seed=episode)
#         total_reward = 0
#         episode_torques = []
#         episode_angles = []
#         episode_times = []

#         for step in range(max_steps):
#             if random.uniform(0, 1) < epsilon:
#                 action = env.action_space.sample()
#             else:
#                 action = controller.act(obs)
#             action = np.clip(action, env.action_space.low, env.action_space.high)

#             obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated

#             total_reward += reward
#             episode_times.append(step)
#             episode_torques.append(action[0])  # Assuming 1D torque
#             episode_angles.append(obs[2])      # Assuming pole angle

#             if done:
#                 if abs(obs[2]) > 0.8:
#                     fall_count += 1
#                 break

#         rewards.append(total_reward)
#         overshoots.append(get_overshoot(episode_angles))
#         settling_times.append(get_settling_time(episode_times, episode_angles))
#         energy.append(get_energy(episode_torques))
#         smoothness.append(get_smoothness(episode_angles, episode_times))

#         epsilon = max(epsilon_min, epsilon * epsilon_decay)

#     convergence_time = get_convergence_time(rewards)
#     avg_reward = get_avg_reward(rewards)
#     fall_rate = get_fall_rate(fall_count, num_episodes)
#     robustness = get_robustness(rewards)

#     metrics = {
#         "Convergence Time (ep)": convergence_time,
#         "Avg Reward": avg_reward,
#         "Overshoot (°)": np.mean(overshoots),
#         "Settling Time (s)": np.mean(settling_times),
#         "Fall Rate (%)": fall_rate,
#         "Energy (∑|τ|)": np.mean(energy),
#         "Smoothness (Jerk)": np.mean(smoothness),
#         "Robustness": robustness
#     }

#     return rewards, metrics, overshoots, settling_times, energy, smoothness

# # 🚀 Main evaluation script
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True, help="Path to config.yaml")
#     parser.add_argument('--model_path', type=str, required=True, help="Path to PPO model (.zip)")
#     parser.add_argument('--use_wandb', action='store_true', help="Log results to wandb")
#     args = parser.parse_args()

#     config = load_config(args.config)

#     if args.use_wandb:
#         wandb.init(
#             project="assistive-walker-eval-DRL",
#             config=vars(config),
#             name=f"eval_PPO_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
#         )

#     env = gymnasium.make("CartPole-v1", render_mode="human")
#     env.action_space.seed(config.seed)
#     env.observation_space.seed(config.seed)
#     np.random.seed(config.seed)

#     model = PPO.load(args.model_path)
#     controller = DRLController(model, "PPO", action_space=env.action_space)

#     logger.info(f"🔍 Evaluating PPO controller...")

#     rewards, metrics, overshoots, settling_times, energy, smoothness = evaluate_controller(env, controller, config.eval_episodes, config.max_steps)
#     mean_reward = np.mean(rewards)

#     logger.info(f"📊 Mean reward over {config.eval_episodes} episodes: {mean_reward}")

#     if args.use_wandb:
#         table = wandb.Table(columns=["Episode", "Reward", "Overshoot", "Settling Time", "Energy", "Smoothness"])
#         for ep in range(config.eval_episodes):
#             table.add_data(ep, rewards[ep], overshoots[ep], settling_times[ep], energy[ep], smoothness[ep])
#             wandb.log({
#                 "episode": ep,
#                 "reward": rewards[ep],
#                 "overshoot": overshoots[ep],
#                 "settling_time": settling_times[ep],
#                 "energy": energy[ep],
#                 "smoothness": smoothness[ep],
#                 "Step": ep,
#             })
#         wandb.log({
#             "mean_reward": mean_reward,
#             **metrics,
#             "Evaluation Episodes": table
#         })
#         wandb.finish()

#     env.close()
