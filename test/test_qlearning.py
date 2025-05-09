import argparse
import gymnasium as gym
import numpy as np
import os
import pickle
import random
import yaml
from types import SimpleNamespace
from datetime import datetime
import wandb
from gymnasium.envs.registration import register
import sys

# 📁 Local imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.cartpole import CartPoleDiscreteEnv
from controllers.qlearning_controller import QLearningController
from train.utils.logger import setup_logger

logger = setup_logger()

# ✅ Register custom environment
register(
    id='CartPole-v1',
    entry_point='environments.cartpole:CartPoleDiscreteEnv',
)

# 🧾 Config loader
def load_config(config_path):
    with open(config_path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

# 📈 Metrics functions (same as DRL)
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

def get_fall_rate(falls, total):
    return float((falls / total) * 100)

def get_energy(torques):
    flat = [abs(x) for vec in torques for x in vec]
    return float(sum(flat))

def get_smoothness(angles, steps):
    if len(steps) > 1:
        jerks = np.diff(angles) / np.diff(steps)
        return float(np.mean(np.abs(jerks)))
    return 0.0

def get_robustness(rewards):
    return float(np.std(rewards))

def evaluate_q_agent(env, agent, num_episodes, max_steps):
    rewards, overshoots, settling_times = [], [], []
    energy_list, smoothness_list = [], []
    fall_count = 0

    bins = [
        np.linspace(-2.4, 2.4, 10),  # Cart position
        np.linspace(-3.0, 3.0, 10),  # Cart velocity
        np.linspace(-0.2, 0.2, 10),  # Pole angle
        np.linspace(-2.0, 2.0, 10),  # Pole velocity
    ]

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)

        # # Discretize state based on the bins
        # state = [np.digitize(obs[i], bins[i]) for i in range(len(obs))]
        # state = np.array(state)
        # q_table_shape = np.array(agent.q_table.shape[:-1])
          # Discretize state based on the bins
        state = [np.digitize(obs[i], bins[i]) for i in range(len(obs))]
        # state = tuple(state)  # Convert to tuple for dictionary key
        q_table_shape = (len(bins), 2) 
        # next_state = list(state)  # Initialize next_state as a list
        # Initializing next_state after first discretization
        next_state = state
        min_bound = np.zeros_like(next_state)
        max_bound = q_table_shape - 1
        state_idx = tuple(np.clip(next_state, min_bound, max_bound).astype(int))

        # Debug info
        print("🔍 Debug Info:")
        print("State:", state)
        print("State shape:", state.shape)
        print("Q-table shape (full):", agent.q_table.shape)
        print("Q-table state shape (excluding action dim):", q_table_shape)
        print("Min bound shape:", np.zeros_like(state).shape)
        print("Max bound shape:", (q_table_shape - 1).shape)

        total_reward = 0
        torques, angles, steps = [], [], []

        for step in range(max_steps):
            # Ensure that state_idx dimensions match q_table's shape
            if len(state_idx) != len(agent.q_table.shape[:-1]):
                print(f"Dimension mismatch: state_idx {state_idx}, q_table.shape {agent.q_table.shape}")
                break  # Exit early for this episode

            action = int(np.argmax(agent.q_table[state_idx]))
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Update next_state
            next_state = [np.digitize(next_obs[i], bins[i]) for i in range(len(next_obs))]
            next_state = np.array(next_state)
            # next_state = list(next_state)


            # Ensure state_idx is updated after next_state calculation
            state_idx = tuple(np.clip(next_state, min_bound, max_bound).astype(int))

            torques.append([action])  # Discrete torque representation
            angles.append(next_obs[2])  # Record the pole angle
            steps.append(step)  # Record the step
            total_reward += reward  # Add the reward

            if done:
                if abs(next_obs[2]) > 0.8:  # Fall detection
                    fall_count += 1
                break  # Exit if done
        rewards.append(total_reward)
        # Only compute these metrics if angles has been populated
        if angles:
            overshoots.append(get_overshoot(angles))
            settling_times.append(get_settling_time(steps, angles))
            energy_list.append(get_energy(torques))
            smoothness_list.append(get_smoothness(angles, steps))
        else:
            overshoots.append(0)
            settling_times.append(max_steps)
            energy_list.append(0)
            smoothness_list.append(0)

    # Calculate the metrics
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





# 🚀 Main
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)
    env = CartPoleDiscreteEnv(render_mode=True)
        # Initialize a random q_table for a discrete state space
    q_table = {}
    for cart_pos in range(10):
        for cart_vel in range(10):
            for pole_angle in range(10):
                for pole_vel in range(10):
                    state = (cart_pos, cart_vel, pole_angle, pole_vel)
                    q_table[state] = [0] * 2  # Two possible actions

# Now, initialize the QLearningController with the q_table
        agent = QLearningController(q_table)
     # Evaluate the agent
    rewards, metrics, overshoots, settling_times, energy_list, smoothness_list = evaluate_q_agent(
        env,agent,  num_episodes=100, max_steps=200
    )
  
    if args.use_wandb:
        # 💾 Combine all evaluation data
            data = []
            for ep in range(len(rewards)):
                data.append([
                    ep,
                    float(rewards[ep]),
                    float(overshoots[ep]),
                    float(settling_times[ep]),
                    float(energy_list[ep]),
                    float(smoothness_list[ep]),
                ])

            # 🧾 Create W&B Table
            table = wandb.Table(data=data, columns=[
                "episode", "reward", "overshoot", "settling_time", "energy", "smoothness"
            ])

            # 📊 Log metrics and plots
            wandb.log({
                "mean_reward": float(np.mean(rewards)),
                **{k: float(v) for k, v in metrics.items()},
                "Reward Curve": wandb.plot.line(table, "episode", "reward", title="Reward per Episode"),
                "Overshoot Curve": wandb.plot.line(table, "episode", "overshoot", title="Overshoot per Episode"),
                "Settling Time Curve": wandb.plot.line(table, "episode", "settling_time", title="Settling Time per Episode"),
                "Energy Curve": wandb.plot.line(table, "episode", "energy", title="Energy per Episode"),
                "Smoothness Curve": wandb.plot.line(table, "episode", "smoothness", title="Smoothness per Episode"),
                "evaluation_table": table
            })
            wandb.finish()


    # if args.use_wandb:
    #     wandb.init(
    #         project="assistive-walker-eval-Q",
    #         config=vars(config),
    #         name=f"eval_Q_{datetime.now():%Y%m%d_%H%M%S}",
    #     )

    # # Load Q-table
    # with open(args.model_path, 'rb') as f:
    #     q_table = pickle.load(f)
    # controller = QLearningController(q_table)

    # env = gym.make("CartPole-v1", render_mode="human")
    # env.action_space.seed(config.seed)
    # env.observation_space.seed(config.seed)
    # np.random.seed(config.seed)
    # random.seed(config.seed)

    # logger.info("🧠 Evaluating Q-learning agent...")

    # rewards, metrics, overshoots, settling_times, energy_list, smoothness_list = \
    #     evaluate_q_agent(env, controller, config.eval_episodes, config.max_steps)

    # logger.info(f"📊 Evaluation Complete. Mean Reward: {np.mean(rewards):.2f}")
    # for k, v in metrics.items():
    #     logger.info(f"{k}: {v:.4f}")

    # if args.use_wandb:
    #     data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
    #     table = wandb.Table(data=data, columns=["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"])

    #     wandb.log({"mean_reward": np.mean(rewards), **metrics})
    #     wandb.log({
    #         "Reward Curve": wandb.plot.line(table, "episode", "reward", title="Reward per Episode"),
    #         "Overshoot Curve": wandb.plot.line(table, "episode", "overshoot", title="Overshoot per Episode"),
    #         "Settling Time Curve": wandb.plot.line(table, "episode", "settling_time", title="Settling Time per Episode"),
    #         "Energy Curve": wandb.plot.line(table, "episode", "energy", title="Energy per Episode"),
    #         "Smoothness Curve": wandb.plot.line(table, "episode", "smoothness", title="Smoothness per Episode"),
    #         "evaluation_table": table
    #     })
    #     wandb.finish()

    # env.close()


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

# # 📁 Local imports
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
# from environments.cartpole import CartPoleDiscreteEnv
# from controllers.qlearning_controller import QLearningController
# from train.utils.logger import setup_logger

# logger = setup_logger()

# # ✅ Register custom environment
# register(
#     id='CartPole-v1',
#     entry_point='environments.cartpole:CartPoleDiscreteEnv',
# )

# # 🧾 Config loader
# def load_config(config_path):
#     with open(config_path, 'r') as f:
#         return SimpleNamespace(**yaml.safe_load(f))

# # 📈 Metrics functions (same as DRL)
# def get_convergence_time(rewards, threshold=0.95):
#     for episode, reward in enumerate(rewards):
#         if reward >= threshold:
#             return episode
#     return len(rewards)

# def get_avg_reward(rewards):
#     return float(np.mean(rewards))

# def get_overshoot(angles, target_angle=0.0):
#     return float(max(abs(a - target_angle) for a in angles))

# def get_settling_time(times, angles, target_angle=0.0, tolerance=0.05):
#     for t, a in zip(times, angles):
#         if abs(a - target_angle) <= tolerance:
#             return float(t)
#     return float(max(times))

# def get_fall_rate(falls, total):
#     return float((falls / total) * 100)

# def get_energy(torques):
#     flat = [abs(x) for vec in torques for x in vec]
#     return float(sum(flat))

# def get_smoothness(angles, steps):
#     if len(steps) > 1:
#         jerks = np.diff(angles) / np.diff(steps)
#         return float(np.mean(np.abs(jerks)))
#     return 0.0

# def get_robustness(rewards):
#     return float(np.std(rewards))

# def evaluate_q_agent(env, agent, num_episodes, max_steps):
#     rewards, overshoots, settling_times = [], [], []
#     energy_list, smoothness_list = [], []
#     fall_count = 0

#     for ep in range(num_episodes):
#         obs, _ = env.reset(seed=ep)
#         state = tuple(np.digitize(obs, bins=np.linspace(-1, 1, 10)))  # Discretizing state
#         total_reward = 0
#         torques, angles, steps = [], [], []

#         for step in range(max_steps):
#             # Ensure that the state has the correct dimensionality for q_table access
#             state_idx = tuple(np.clip(state, 0, np.array(agent.q_table.shape) - 1))  # Prevent out of bounds
#             if len(state_idx) == len(agent.q_table.shape):  # Ensure matching dimensionality
#                 if state_idx in agent.q_table:
#                     action = int(np.argmax(agent.q_table[state_idx]))
#                 else:
#                     action = env.action_space.sample()

#                 next_obs, reward, terminated, truncated, info = env.step(action)
#                 done = terminated or truncated
#                 next_state = tuple(np.digitize(next_obs, bins=np.linspace(-1, 1, 10)))

#                 torques.append([action])  # Discrete torque representation
#                 angles.append(next_obs[2])
#                 steps.append(step)
#                 total_reward += reward
#                 state = next_state

#                 if done:
#                     if abs(next_obs[2]) > 0.8:
#                         fall_count += 1
#                     break
#             else:
#                 logger.warning(f"State dimension mismatch: {len(state_idx)} vs {len(agent.q_table.shape)}")

#         rewards.append(total_reward)
#         overshoots.append(get_overshoot(angles))
#         settling_times.append(get_settling_time(steps, angles))
#         energy_list.append(get_energy(torques))
#         smoothness_list.append(get_smoothness(angles, steps))

#     metrics = {
#         "Convergence Time (ep)": get_convergence_time(rewards),
#         "Avg Reward": get_avg_reward(rewards),
#         "Overshoot (°)": float(np.mean(overshoots)),
#         "Settling Time (steps)": float(np.mean(settling_times)),
#         "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
#         "Energy (∑|τ|)": float(np.mean(energy_list)),
#         "Smoothness (Jerk)": float(np.mean(smoothness_list)),
#         "Robustness": get_robustness(rewards)
#     }

#     return rewards, metrics, overshoots, settling_times, energy_list, smoothness_list



# # # 🧪 Evaluation
# # def evaluate_q_agent(env, agent, num_episodes, max_steps):
# #     rewards, overshoots, settling_times = [], [], []
# #     energy_list, smoothness_list = [], []
# #     fall_count = 0

# #     for ep in range(num_episodes):
# #         obs, _ = env.reset(seed=ep)
# #         state = tuple(np.digitize(obs, bins=np.linspace(-1, 1, 10)))  # or use your own state discretizer
# #         total_reward = 0
# #         torques, angles, steps = [], [], []

# #         for step in range(max_steps):
# #             if state in agent.q_table:
# #                 action = int(np.argmax(agent.q_table[state]))
# #             else:
# #                 action = env.action_space.sample()

# #             next_obs, reward, terminated, truncated, info = env.step(action)
# #             done = terminated or truncated
# #             next_state = tuple(np.digitize(next_obs, bins=np.linspace(-1, 1, 10)))

# #             torques.append([action])  # discrete torque representation
# #             angles.append(next_obs[2])
# #             steps.append(step)
# #             total_reward += reward
# #             state = next_state

# #             if done:
# #                 if abs(next_obs[2]) > 0.8:
# #                     fall_count += 1
# #                 break

# #         rewards.append(total_reward)
# #         overshoots.append(get_overshoot(angles))
# #         settling_times.append(get_settling_time(steps, angles))
# #         energy_list.append(get_energy(torques))
# #         smoothness_list.append(get_smoothness(angles, steps))

# #     metrics = {
# #         "Convergence Time (ep)": get_convergence_time(rewards),
# #         "Avg Reward": get_avg_reward(rewards),
# #         "Overshoot (°)": float(np.mean(overshoots)),
# #         "Settling Time (steps)": float(np.mean(settling_times)),
# #         "Fall Rate (%)": get_fall_rate(fall_count, num_episodes),
# #         "Energy (∑|τ|)": float(np.mean(energy_list)),
# #         "Smoothness (Jerk)": float(np.mean(smoothness_list)),
# #         "Robustness": get_robustness(rewards)
# #     }

# #     return rewards, metrics, overshoots, settling_times, energy_list, smoothness_list

# # 🚀 Main
# if __name__ == "__main__":
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--config', type=str, required=True)
#     parser.add_argument('--model_path', type=str, required=True)
#     parser.add_argument('--use_wandb', action='store_true')
#     args = parser.parse_args()

#     config = load_config(args.config)

#     if args.use_wandb:
#         wandb.init(
#             project="assistive-walker-eval-Q",
#             config=vars(config),
#             name=f"eval_Q_{datetime.now():%Y%m%d_%H%M%S}",
#         )

#     # Load Q-table
#     with open(args.model_path, 'rb') as f:
#         q_table = pickle.load(f)
#     controller = QLearningController(q_table)

#     env = gym.make("CartPole-v1", render_mode="human")
#     env.action_space.seed(config.seed)
#     env.observation_space.seed(config.seed)
#     np.random.seed(config.seed)
#     random.seed(config.seed)

#     logger.info("🧠 Evaluating Q-learning agent...")

#     rewards, metrics, overshoots, settling_times, energy_list, smoothness_list = \
#         evaluate_q_agent(env, controller, config.eval_episodes, config.max_steps)

#     logger.info(f"📊 Evaluation Complete. Mean Reward: {np.mean(rewards):.2f}")
#     for k, v in metrics.items():
#         logger.info(f"{k}: {v:.4f}")

#     if args.use_wandb:
#         data = list(zip(range(len(rewards)), rewards, overshoots, settling_times, energy_list, smoothness_list))
#         table = wandb.Table(data=data, columns=["episode", "reward", "overshoot", "settling_time", "energy", "smoothness"])

#         wandb.log({"mean_reward": np.mean(rewards), **metrics})
#         wandb.log({
#             "Reward Curve": wandb.plot.line(table, "episode", "reward", title="Reward per Episode"),
#             "Overshoot Curve": wandb.plot.line(table, "episode", "overshoot", title="Overshoot per Episode"),
#             "Settling Time Curve": wandb.plot.line(table, "episode", "settling_time", title="Settling Time per Episode"),
#             "Energy Curve": wandb.plot.line(table, "episode", "energy", title="Energy per Episode"),
#             "Smoothness Curve": wandb.plot.line(table, "episode", "smoothness", title="Smoothness per Episode"),
#             "evaluation_table": table
#         })
#         wandb.finish()

#     env.close()
