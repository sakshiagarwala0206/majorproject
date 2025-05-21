import argparse
import gymnasium as gym
import numpy as np
import pickle
import wandb
from datetime import datetime
from gymnasium.envs.registration import register
import os
import sys
import yaml
# Project-specific imports
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
from environments.walker_1 import AssistiveWalkerDiscreteEnv

register(
    id="AssistiveWalkerDiscreteEnv-v0",
    entry_point="environments.walker_1:AssistiveWalkerDiscreteEnv",
    max_episode_steps=10000,
)

class QLearningController:
    def __init__(self, q_table, bins, obs_low, obs_high):
        self.q_table = q_table
        self.bins = np.array(bins)
        self.obs_low = np.array(obs_low)
        self.obs_high = np.array(obs_high)
        
    def discretize_state(self, state):
        ratios = (state - self.obs_low) / (self.obs_high - self.obs_low + 1e-8)
        ratios = np.clip(ratios, 0, 0.9999)
        return tuple((ratios * self.bins).astype(int))
    
    def predict(self, obs):
        discrete_state = self.discretize_state(obs[:7])  # Use first 7 obs elements
        return np.argmax(self.q_table[discrete_state]), None

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def evaluate_controller(env, controller, episodes, max_steps):
    metrics = {
        'rewards': [],
        'overshoots': [],
        'settling_times': [],
        'energy': [],
        'smoothness': [],
        'fall_count': 0,
        'imu_data': []
    }

    for ep in range(episodes):
        obs, _ = env.reset()
        total_reward = 0
        step_data = {
            'angles': [],
            'torques': [],
            'steps': [],
            'imu': []
        }

        for step in range(max_steps):
            action, _ = controller.predict(obs)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            
            # Log IMU data (indices 7-15)
            if len(next_obs) >= 16:
                imu_entry = {
                    'roll': next_obs[7],
                    'pitch': next_obs[8],
                    'yaw': next_obs[9],
                    'ang_vel_x': next_obs[10],
                    'ang_vel_y': next_obs[11],
                    'ang_vel_z': next_obs[12],
                    'lin_acc_x': next_obs[13],
                    'lin_acc_y': next_obs[14],
                    'lin_acc_z': next_obs[15],
                    'step': step,
                    'episode': ep
                }
                metrics['imu_data'].append(imu_entry)
                if wandb.run:
                    wandb.log(imu_entry)

            # Update metrics
            total_reward += reward
            step_data['angles'].append(next_obs[0])
            step_data['torques'].append(action)
            step_data['steps'].append(step)
            
            if terminated or truncated:
                if abs(next_obs[0]) > 0.5:
                    metrics['fall_count'] += 1
                break

        # Calculate episode metrics
        metrics['rewards'].append(total_reward)
        metrics['overshoots'].append(max(step_data['angles']) if step_data['angles'] else 0)
        metrics['settling_times'].append(_calculate_settling_time(step_data['angles'], step_data['steps']))
        metrics['energy'].append(np.sum(np.abs(step_data['torques'])))
        metrics['smoothness'].append(_calculate_smoothness(step_data['angles'], step_data['steps']))

    return metrics

def _calculate_settling_time(angles, steps, threshold=0.1):
    if not angles:
        return 0.0
    target = angles[0]
    for i, angle in enumerate(angles):
        if abs(angle - target) <= threshold:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0

def _calculate_smoothness(angles, steps):
    if len(steps) < 2:
        return 0.0
    jerks = np.diff(angles) / np.diff(steps)
    return float(np.mean(np.abs(jerks)))

def log_to_wandb(metrics, config):
    # Log summary metrics
    summary = {
        'avg_reward': np.mean(metrics['rewards']),
        'fall_rate': metrics['fall_count'] / len(metrics['rewards']),
        'avg_energy': np.mean(metrics['energy']),
        'avg_smoothness': np.mean(metrics['smoothness']),
        'avg_overshoot': np.mean(metrics['overshoots']),
        'avg_settling_time': np.mean(metrics['settling_times'])
    }
    wandb.log(summary)
    
    # Log raw data table
    table = wandb.Table(columns=["Episode", "Reward", "Overshoot", "Settling Time", "Energy", "Smoothness"])
    for ep in range(len(metrics['rewards'])):
        table.add_data(ep, metrics['rewards'][ep], metrics['overshoots'][ep],
                      metrics['settling_times'][ep], metrics['energy'][ep], metrics['smoothness'][ep])
    wandb.log({"performance_table": table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained Q-table')
    parser.add_argument('--use_wandb', action='store_true', help='Enable wandb logging')
    args = parser.parse_args()

    config = load_config(args.config)
    
    # Load Q-table
    with open(args.model_path, 'rb') as f:
        q_table = pickle.load(f)

    # Initialize environment and controller
    env = gym.make("AssistiveWalkerDiscreteEnv-v0", render_mode="human")
    controller = QLearningController(
        q_table=q_table,
        bins=config['bins'],
        obs_low=config['obs_low'],
        obs_high=config['obs_high']
    )

    # WandB initialization
    if args.use_wandb:
        wandb.init(
            project="assistive-walker-q-test",
            config=config,
            name=f"q_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

    # Run evaluation
    metrics = evaluate_controller(
        env=env,
        controller=controller,
        episodes=config.get('eval_episodes', 30),
        max_steps=config.get('max_steps', 1000)
    )

    # Final logging
    if args.use_wandb:
        log_to_wandb(metrics, config)
        wandb.finish()

    env.close()
