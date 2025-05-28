


# import argparse
# import gymnasium as gym
# import numpy as np
# import os
# import pickle
# import random
# import yaml
# from types import SimpleNamespace
# from datetime import datetime
# import wandb
# from gymnasium.envs.registration import register
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from environments.cartpole import CartPoleDiscreteEnv
# from controllers.qlearning_controller import QLearningController
# from train.utils.logger import setup_logger

# logger = setup_logger()

# register(
#     id='CustomCartPole',
#     entry_point='environments.cartpole:CartPoleDiscreteEnv',
# )

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return SimpleNamespace(**yaml.safe_load(f))

# def get_convergence_time(rewards, threshold=0.95):
#     for episode, reward in enumerate(rewards):
#         if reward >= threshold:
#             return episode
#     return len(rewards)

# def get_avg_reward(rewards):
#     return float(np.mean(rewards))

# def get_overshoot(angles, target_angle=0.0):
#     return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
#     for t, a in zip(times, angles):
#         if abs(a - target_angle) <= tolerance:
#             return float(t)
#     return float(max(times)) if times else 0.0

# def get_fall_rate(falls, total):
#     return float((falls / total) * 100)

# def get_energy(forces):
#     flat = [abs(x) for x in forces]
#     return float(sum(flat))

# def get_smoothness(values, steps):
#     if len(steps) > 1:
#         jerks = np.diff(values) / np.diff(steps)
#         return float(np.mean(np.abs(jerks)))
#     return 0.0

# def get_robustness(rewards):
#     return float(np.std(rewards))

# def evaluate_q_agent(env, agent, num_episodes, max_steps):
#     rewards, overshoots, settling_times = [], [], []
#     energy_list, smoothness_list = [], []
#     fall_count = 0

#     # For logging all obs and actions
#     all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions = [], [], [], [], []

#     bins = [
#         np.linspace(-2.4, 2.4, 10),  # Cart position
#         np.linspace(-3.0, 3.0, 10),  # Cart velocity
#         np.linspace(-0.2, 0.2, 10),  # Pole angle
#         np.linspace(-2.0, 2.0, 10),  # Pole velocity
#     ]

#     # DEBUG: Print Q-table type for reproducibility
#     print(f"DEBUG: Q-table type is {type(agent.q_table)}")

#     # For dict Q-tables, get state tuple length from keys
#     if isinstance(agent.q_table, dict):
#         sample_key = next(iter(agent.q_table))
#         qtable_state_dim = len(sample_key)
#     elif hasattr(agent.q_table, 'shape'):
#         qtable_state_dim = len(agent.q_table.shape) - 1
#     else:
#         raise RuntimeError("Unknown Q-table type")

#     for ep in range(num_episodes):
#         obs, _ = env.reset(seed=ep)
#         state = [np.digitize(obs[i], bins[i]) for i in range(len(obs))]
#         num_bins = [len(b) for b in bins]
#         min_bound = np.zeros(len(num_bins))
#         max_bound = np.array(num_bins) - 1

#         state_idx = tuple(np.clip(state, min_bound, max_bound).astype(int))

#         total_reward = 0
#         actions, cart_pos, cart_vel, pole_angle, pole_vel, steps = [], [], [], [], [], []

#         for step in range(max_steps):
#             # Robust dimension check for Q-table
#             if len(state_idx) != qtable_state_dim:
#                 print(f"DEBUG: State index length {len(state_idx)} does not match Q-table state dimension {qtable_state_dim}")
#                 break

#             # Robust Q-value lookup for both array and dict
#             if isinstance(agent.q_table, dict):
#                 q_values = agent.q_table.get(state_idx, np.zeros(env.action_space.n))
#                 if state_idx not in agent.q_table:
#                     print(f"DEBUG: Unseen state encountered: {state_idx}")
#             else:
#                 q_values = agent.q_table[state_idx]
#             action = int(np.argmax(q_values))

#             next_obs, reward, terminated, truncated, info = env.step(action)
#             done = terminated or truncated

#             cart_pos.append(next_obs[0])
#             cart_vel.append(next_obs[1])
#             pole_angle.append(next_obs[2])
#             pole_vel.append(next_obs[3])
#             actions.append(action)
#             steps.append(step)

#             next_state = [np.digitize(next_obs[i], bins[i]) for i in range(len(next_obs))]
#             next_state = np.array(next_state)
#             state_idx = tuple(np.clip(next_state, min_bound, max_bound).astype(int))

#             total_reward += reward

#             if done:
#                 if abs(next_obs[2]) > 0.8:
#                     fall_count += 1
#                 break

#         all_cart_pos.append(cart_pos)
#         all_cart_vel.append(cart_vel)
#         all_pole_angle.append(pole_angle)
#         all_pole_vel.append(pole_vel)
#         all_actions.append(actions)

#         rewards.append(total_reward)
#         overshoots.append(get_overshoot(pole_angle) if pole_angle else 0)
#         settling_times.append(get_settling_time(steps, pole_angle) if pole_angle else max_steps)
#         energy_list.append(get_energy(actions))
#         smoothness_list.append(get_smoothness(pole_angle, steps))

#     def ep_stats(series):
#         return {
#             "mean": float(np.mean([np.mean(ep) if ep else 0 for ep in series])),
#             "max": float(np.max([np.max(ep) if ep else 0 for ep in series])),
#             "min": float(np.min([np.min(ep) if ep else 0 for ep in series])),
#             "std": float(np.std([np.std(ep) if ep else 0 for ep in series]))
#         }

#     metrics = {
#         "Convergence Time (ep)": get_convergence_time(rewards),
#         "Avg Reward": get_avg_reward(rewards),
#         "Overshoot (°)": float(np.mean(overshoots)),
#         "Settling Time (steps)": float(np.mean(settling_times)),
#         "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
#         "Energy (∑|F|)": float(np.mean(energy_list)),
#         "Smoothness (Jerk)": float(np.mean(smoothness_list)),
#         "Robustness": get_robustness(rewards),
#         "CartPos Mean": ep_stats(all_cart_pos)["mean"],
#         "CartPos Max": ep_stats(all_cart_pos)["max"],
#         "CartPos Min": ep_stats(all_cart_pos)["min"],
#         "CartPos Std": ep_stats(all_cart_pos)["std"],
#         "CartVel Mean": ep_stats(all_cart_vel)["mean"],
#         "CartVel Max": ep_stats(all_cart_vel)["max"],
#         "CartVel Min": ep_stats(all_cart_vel)["min"],
#         "CartVel Std": ep_stats(all_cart_vel)["std"],
#         "PoleAngle Mean": ep_stats(all_pole_angle)["mean"],
#         "PoleAngle Max": ep_stats(all_pole_angle)["max"],
#         "PoleAngle Min": ep_stats(all_pole_angle)["min"],
#         "PoleAngle Std": ep_stats(all_pole_angle)["std"],
#         "PoleVel Mean": ep_stats(all_pole_vel)["mean"],
#         "PoleVel Max": ep_stats(all_pole_vel)["max"],
#         "PoleVel Min": ep_stats(all_pole_vel)["min"],
#         "PoleVel Std": ep_stats(all_pole_vel)["std"],
#         "Action Mean": ep_stats(all_actions)["mean"],
#         "Action Max": ep_stats(all_actions)["max"],
#         "Action Min": ep_stats(all_actions)["min"],
#         "Action Std": ep_stats(all_actions)["std"],
#     }

#     return rewards, metrics, overshoots, settling_times, energy_list, smoothness_list, all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions

# # 🚀 Main
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)
#     env = CartPoleDiscreteEnv(render_mode=True, test_mode=True)

#     # Set seeds for reproducibility
#     env.action_space.seed(getattr(config, 'seed', 42))
#     env.observation_space.seed(getattr(config, 'seed', 42))
#     np.random.seed(getattr(config, 'seed', 42))
#     random.seed(getattr(config, 'seed', 42))

#     # Load Q-table
#     with open(args.model_path, "rb") as f:
#         q_table = pickle.load(f)
#     agent = QLearningController(q_table)

#     if args.use_wandb:
#         wandb.init(
#             project="Cartpole_test_final",
#             config=vars(config),
#             name=f"eval_Qlearning_{datetime.now():%Y%m%d_%H%M%S}",
#         )

#     num_episodes = getattr(config, "eval_episodes", 100)
#     max_steps = getattr(config, "max_steps", 1000)

#     results = evaluate_q_agent(env, agent, num_episodes=num_episodes, max_steps=max_steps)
#     rewards, metrics, overshoots, settling_times, energy_list, smoothness_list, all_cart_pos, all_cart_vel, all_pole_angle, all_pole_vel, all_actions = results

#     print("==== Observation and Action Statistics ====")
#     for k in ["CartPos", "CartVel", "PoleAngle", "PoleVel", "Action"]:
#         print(f"{k} Mean: {metrics[k+' Mean']:.3f}, Max: {metrics[k+' Max']:.3f}, Min: {metrics[k+' Min']:.3f}, Std: {metrics[k+' Std']:.3f}")

#     if args.use_wandb:
#         data = []
#         for ep in range(len(rewards)):
#             data.append([
#                 ep,
#                 float(rewards[ep]),
#                 float(overshoots[ep]),
#                 float(settling_times[ep]),
#                 float(energy_list[ep]),
#                 float(smoothness_list[ep]),
#                 float(np.mean(all_cart_pos[ep]) if all_cart_pos[ep] else 0),
#                 float(np.mean(all_cart_vel[ep]) if all_cart_vel[ep] else 0),
#                 float(np.mean(all_pole_angle[ep]) if all_pole_angle[ep] else 0),
#                 float(np.mean(all_pole_vel[ep]) if all_pole_vel[ep] else 0),
#                 float(np.mean(all_actions[ep]) if all_actions[ep] else 0),
#             ])
#         columns = [
#             "episode", "reward", "overshoot", "settling_time", "energy", "smoothness",
#             "mean_cart_pos", "mean_cart_vel", "mean_pole_angle", "mean_pole_vel", "mean_action"
#         ]
#         table = wandb.Table(data=data, columns=columns)
#         wandb.log({
#             "mean_reward": float(np.mean(rewards)),
#             **{k: float(v) for k, v in metrics.items()},
#             "Reward Curve": wandb.plot.line(table, "episode", "reward", title="Reward per Episode"),
#             "Cart Position Curve": wandb.plot.line(table, "episode", "mean_cart_pos", title="Cart Position per Episode"),
#             "Cart Velocity Curve": wandb.plot.line(table, "episode", "mean_cart_vel", title="Cart Velocity per Episode"),
#             "Pole Angle Curve": wandb.plot.line(table, "episode", "mean_pole_angle", title="Pole Angle per Episode"),
#             "Pole Velocity Curve": wandb.plot.line(table, "episode", "mean_pole_vel", title="Pole Velocity per Episode"),
#             "Action Curve": wandb.plot.line(table, "episode", "mean_action", title="Action per Episode"),
#             "Overshoot Curve": wandb.plot.line(table, "episode", "overshoot", title="Overshoot per Episode"),
#             "Settling Time Curve": wandb.plot.line(table, "episode", "settling_time", title="Settling Time per Episode"),
#             "Energy Curve": wandb.plot.line(table, "episode", "energy", title="Energy per Episode"),
#             "Smoothness Curve": wandb.plot.line(table, "episode", "smoothness", title="Smoothness per Episode"),
#             "evaluation_table": table
#         })
#         wandb.finish()


# import argparse
# import gymnasium as gym
# import numpy as np
# import pickle
# import os
# import random
# import yaml
# from types import SimpleNamespace
# from datetime import datetime
# import wandb
# from gymnasium.envs.registration import register
# import sys

# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from environments.cartpole import CartPoleDiscreteEnv
# register(
#     id='CustomCartPole-v0',
#     entry_point='environments.cartpole:CartPoleDiscreteEnv',
#     max_episode_steps=500,
# )

# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         config_dict = yaml.safe_load(f)
#     return SimpleNamespace(**config_dict)

# def get_statistics(data):
#     return {
#         "mean": float(np.mean(data)),
#         "std": float(np.std(data)),
#         "median": float(np.median(data))
#     }

# def get_fall_rate(falls, total):
#     return float((falls / total) * 100)

# def get_energy(forces):
#     return float(np.sum(np.abs(forces)))

# def get_smoothness(angles, steps):
#     # Only compute if at least 2 steps, else return 0
#     if len(angles) > 1 and len(steps) > 1:
#         jerks = np.diff(angles) / np.diff(steps)
#         if len(jerks) > 0 and not np.any(np.isnan(jerks)):
#             return float(np.mean(np.abs(jerks)))
#     return 0.0

# def get_overshoot(angles, target_angle=0.0):
#     return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(times, angles, disturbance_step=200, tolerance=0.05):
#     # Settling time after disturbance, always non-negative
#     settled = False
#     for t, a in zip(times, angles):
#         if t >= disturbance_step and abs(a) <= tolerance:
#             settled = True
#             return float(max(0, t - disturbance_step))
#     if not settled:
#         return float(max(times) - disturbance_step) if times else 0.0

# def evaluate_q_agent(env, q_table, bins, num_episodes, max_steps):
#     rewards, overshoots, settling_times, energy_list, smoothness_list = [], [], [], [], []
#     fall_count = 0

#     for ep in range(num_episodes):
#         obs, _ = env.reset(seed=ep)
#         state = tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))
#         total_reward = 0
#         angle_history = []
#         action_history = []
#         steps = []
#         done = False
#         fall_flag = False

#         for step in range(max_steps):
#             # Disturbance: Apply lateral impulse at step 200 if using PyBullet
#             if step == 200 and hasattr(env, "cartpole_id"):
#                 import pybullet as p
#                 p.applyExternalForce(
#                     objectUniqueId=env.cartpole_id,
#                     linkIndex=-1,
#                     forceObj=[4, 0, 0],
#                     posObj=[0, 0, 0],
#                     flags=p.WORLD_FRAME
#                 )

#             # Q-table action selection
#             if state in q_table:
#                 action = int(np.argmax(q_table[state]))
#             else:
#                 action = env.action_space.sample()

#             obs, reward, terminated, truncated, _ = env.step(action)
#             done = terminated or truncated

#             angle_history.append(obs[2])
#             action_history.append(action)
#             steps.append(step)
#             total_reward += reward

#             # Update state
#             state = tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))

#             if done:
#                 # Count as fall if episode ended early (not max_steps)
#                 if step < max_steps - 1:
#                     fall_flag = True
#                     fall_count += 1
#                 break

#         rewards.append(total_reward)
#         energy_list.append(get_energy(action_history))
#         smoothness_list.append(get_smoothness(angle_history, steps))
#         overshoots.append(get_overshoot(angle_history))
#         settling_times.append(get_settling_time(steps, angle_history))

#     metrics = {
#         "Avg Reward": get_statistics(rewards),
#         "Fall Rate (%)": {
#             "mean": get_fall_rate(fall_count, num_episodes),
#             "std": 0.0,
#             "median": get_fall_rate(fall_count, num_episodes)
#         },
#         "Energy (∑|τ|)": get_statistics(energy_list),
#         "Smoothness (Jerk)": get_statistics(smoothness_list),
#         "Overshoot": get_statistics(overshoots),
#         "Settling Time": get_statistics(settling_times)
#     }

#     print("\n==== CartPole Q-learning Evaluation Summary ====")
#     for metric, stats in metrics.items():
#         print(f"{metric}: mean={stats['mean']:.2f}, std={stats['std']:.2f}, median={stats['median']:.2f}")
#     print("===============================================\n")

#     return rewards, overshoots, settling_times, energy_list, smoothness_list, metrics

# def log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config):
#     data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
#     cols = ["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"]
#     episode_table = wandb.Table(data=data, columns=cols)

#     wandb.log({
#         "mean_reward": metrics["Avg Reward"]["mean"],
#         "std_reward": metrics["Avg Reward"]["std"],
#         "median_reward": metrics["Avg Reward"]["median"],
#         "fall_rate_mean": metrics["Fall Rate (%)"]["mean"],
#         "energy_mean": metrics["Energy (∑|τ|)"]["mean"],
#         "smoothness_mean": metrics["Smoothness (Jerk)"]["mean"],
#         "overshoot_mean": metrics["Overshoot"]["mean"],
#         "settling_time_mean": metrics["Settling Time"]["mean"],
#     })

#     plots = {
#         "Reward Curve": ("reward", "Reward per Episode"),
#         "Overshoot Curve": ("overshoot", "Overshoot per Episode"),
#         "Settling Time Curve": ("settling_time", "Settling Time per Episode"),
#         "Energy Curve": ("energy", "Energy per Episode"),
#         "Smoothness Curve": ("smoothness", "Smoothness per Episode"),
#     }

#     for key, (y_col, title) in plots.items():
#         wandb.log({
#             key: wandb.plot.line(episode_table, "episode", y_col, title=title)
#         })

#     wandb.log({"Evaluation Data": episode_table})

# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)

#     if args.use_wandb:
#         wandb.init(
#             project="Cartpole_test_final",
#             config=vars(config),
#             name=f"eval_Qlearning_{datetime.now():%Y%m%d_%H%M%S}",
#         )

#     env = gym.make("CustomCartPole-v0", render_mode="human", test_mode=True)
#     env.action_space.seed(getattr(config, "seed", 0))
#     np.random.seed(getattr(config, "seed", 0))
#     random.seed(getattr(config, "seed", 0))

#     # Load Q-table
#     with open(args.model_path, "rb") as f:
#         q_table = pickle.load(f)

#     # Use same bins as in training
#     bins = [
#         np.linspace(-2.4, 2.4, 10),   # Cart position
#         np.linspace(-3.0, 3.0, 10),   # Cart velocity
#         np.linspace(-0.5, 0.5, 10),   # Pole angle
#         np.linspace(-2.0, 2.0, 10)    # Pole velocity
#     ]

#     rewards, overshoots, settling_times, energy_list, smoothness_list, metrics = evaluate_q_agent(
#         env, q_table, bins, getattr(config, "eval_episodes", 100), getattr(config, "max_steps", 500)
#     )

#     if args.use_wandb:
#         log_to_wandb(rewards, overshoots, settling_times, energy_list, smoothness_list, metrics, config)
#         wandb.finish()

#     env.close()

import argparse
import gymnasium as gym
import numpy as np
import pickle
import os
import random
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb
from gymnasium.envs.registration import register
import sys
import pandas as pd
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole import CartPoleDiscreteEnv

register(
    id='CustomCartPole-v0',
    entry_point='environments.cartpole:CartPoleDiscreteEnv',
    max_episode_steps=500,
)

def load_config(config_path):
    with open(config_path, 'r') as f:
        config_dict = yaml.safe_load(f)
    return SimpleNamespace(**config_dict)

def get_energy(actions):
    return float(np.sum(np.abs(actions)))

def get_smoothness(angles, steps, dt=1.0):
    if len(angles) > 1:
        jerks = np.diff(angles) / (np.diff(steps) * dt)
        return float(np.mean(np.abs(jerks))) if len(jerks) > 0 else 0.0
    return 0.0

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

# def get_settling_time(steps, angles, disturbance_step=200, tolerance=0.05):
#     for t, a in zip(steps, angles):
#         if t >= disturbance_step and abs(a) <= tolerance:
#             return float(max(0, t - disturbance_step))
#     return float(max(steps) - disturbance_step) if steps else 0.0

def get_settling_time(steps, angles, tolerance=0.1):
    """
    Returns the first time (step index) at which the angle enters and remains within the tolerance band.
    If it never settles, returns the episode length.
    """
    for i, a in enumerate(angles):
        if abs(a) <= tolerance:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="Cartpole_test_final",
            config=vars(config),
            name=f"eval_Qlearning_{datetime.now():%Y%m%d_%H%M%S}",
        )

    env = gym.make("CustomCartPole-v0", render_mode="human", test_mode=True)
    env.action_space.seed(getattr(config, "seed", 0))
    np.random.seed(getattr(config, "seed", 0))
    random.seed(getattr(config, "seed", 0))

    with open(args.model_path, "rb") as f:
        q_table = pickle.load(f)

    bins = [
        np.linspace(-2.4, 2.4, 10),   # Cart position
        np.linspace(-3.0, 3.0, 10),   # Cart velocity
        np.linspace(-0.5, 0.5, 10),   # Pole angle
        np.linspace(-2.0, 2.0, 10)    # Pole velocity
    ]

    max_episodes = getattr(config, "eval_episodes", 100)
    max_steps = getattr(config, "max_steps", 500)

    log_data = {
        "episode": [], "reward": [], "overshoot": [], "settling_time": [],
        "smoothness": [], "energy": [], "fall_rate": [], "episode_length": []
    }

    for ep in range(max_episodes):
        obs, _ = env.reset(seed=ep)
        state = tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))
        total_reward = 0
        angle_history = []
        action_history = []
        step_history = []
        done = False
        fall_flag = False
        step = 0

        for step in range(max_steps):
            if step == 200 and hasattr(env, "cartpole_id"):
                import pybullet as p
                p.applyExternalForce(
                    objectUniqueId=env.cartpole_id,
                    linkIndex=-1,
                    forceObj=[4, 0, 0],
                    posObj=[0, 0, 0],
                    flags=p.WORLD_FRAME
                )

            if state in q_table:
                action = int(np.argmax(q_table[state]))
            else:
                action = env.action_space.sample()

            obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            angle_history.append(obs[2])
            action_history.append(action)
            step_history.append(step)
            total_reward += reward

            state = tuple(np.digitize(obs[i], bins[i]) for i in range(len(obs)))

            if done:
                if step < max_steps - 1:
                    fall_flag = True
                break

        overshoot = get_overshoot(angle_history)
        settling_time = get_settling_time(step_history, angle_history)
        smoothness = get_smoothness(angle_history, step_history, dt=1.0)
        energy = get_energy(action_history)
        episode_length = step + 1
        fall_rate = 1 if fall_flag else 0

        log_data["episode"].append(ep)
        log_data["reward"].append(total_reward)
        log_data["overshoot"].append(overshoot)
        log_data["settling_time"].append(settling_time)
        log_data["smoothness"].append(smoothness)
        log_data["energy"].append(energy)
        log_data["fall_rate"].append(fall_rate)
        log_data["episode_length"].append(episode_length)

        if args.use_wandb:
            wandb.log({
                "episode": ep,
                "reward": total_reward,
                "overshoot": overshoot,
                "settling_time": settling_time,
                "smoothness": smoothness,
                "energy": energy,
                "fall_rate": fall_rate,
                "episode_length": episode_length
            })

        print(f"Episode {ep+1}/{max_episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_rate}")

    env.close()

    df = pd.DataFrame(log_data)
    df.to_csv("qlearning_metrics.csv", index=False)
    print("Saved metrics to qlearning_metrics.csv")

    metrics = ["reward", "overshoot", "settling_time", "smoothness", "energy", "fall_rate", "episode_length"]
    episode_table = wandb.Table(dataframe=df) if args.use_wandb else None

    if args.use_wandb:
        # Log summary statistics for each metric
        for metric in metrics:
            wandb.log({
                f"{metric}_mean": float(df[metric].mean()),
                f"{metric}_std": float(df[metric].std()),
                f"{metric}_median": float(df[metric].median())
            })
        # Unified line plots for each metric
        for metric in metrics:
            wandb.log({
                f"{metric}_curve": wandb.plot.line(
                    episode_table, "episode", metric, title=f"{metric.replace('_', ' ').title()} Over Episodes"
                )
            })
        wandb.log({"Evaluation Data": episode_table})
        wandb.finish()
