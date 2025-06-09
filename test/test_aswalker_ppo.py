
import argparse
import gymnasium as gym
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
import pybullet as p
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

from environments.walker import AssistiveWalkerContinuousEnv
from controllers.drl_controller import DRLController
from train.utils.logger import setup_logger
from stable_baselines3 import PPO

logger = setup_logger()

register(
    id="WalkerBalanceContinuousEnv-v0",
    entry_point="environments.walker:AssistiveWalkerContinuousEnv",
    max_episode_steps=1000,
)

def load_config(path):
    with open(path, 'r') as f:
        return SimpleNamespace(**yaml.safe_load(f))

def load_robot():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../'))
    urdf_path = os.path.join(project_root, 'urdf', 'walker.urdf')
    if not os.path.exists(urdf_path):
        raise FileNotFoundError(f"URDF file not found at {urdf_path}")
    robot_id = p.loadURDF(urdf_path, basePosition=[0, 0, 0])
    return robot_id

def get_energy(forces):
    flat = []
    for v in forces:
        try:
            vals = list(v)
        except TypeError:
            vals = [v]
        flat.extend([abs(x) for x in vals])
    return float(sum(flat))

def get_smoothness(angles, steps, dt=1.0):
    if len(angles) > 1 and len(steps) > 1:
        jerks = np.diff(angles) / (np.diff(steps) * dt)
        if len(jerks) > 0 and not np.any(np.isnan(jerks)):
            return float(np.mean(np.abs(jerks)))
    return 0.0

def get_overshoot(angles, target_angle=0.0):
    return float(max(abs(a - target_angle) for a in angles)) if angles else 0.0

def get_settling_time(steps, angles, tolerance=0.1):
    for i, a in enumerate(angles):
        if abs(a) <= tolerance:
            return float(steps[i])
    return float(steps[-1]) if steps else 0.0

def evaluate_controller(env, controller, episodes, max_steps):
    log_data = {
        "episode": [], "reward": [], "overshoot": [], "settling_time": [],
        "smoothness": [], "energy": [], "fall_rate": [], "episode_length": []
    }
    # For penalty breakdown
    penalty_log = {
        "episode": [], "step": [], "angle_penalty": [], "base_penalty": [],
        "energy_penalty": [], "total_reward": []
    }
    fall_count = 0
    robot_id = load_robot()

    for ep in range(episodes):
        obs, _ = env.reset(seed=ep)
        total_reward = 0.0
        episode_forces = []
        episode_angles = []
        episode_steps = []
        done = False
        fall_flag = False

        for step in range(max_steps):
            # Disturbances
            # if step == 100:
            #     base_mass = p.getDynamicsInfo(robot_id, -1)[0]
            #     p.changeDynamics(robot_id, -1, mass=base_mass + 0.3)
            if step == 200:
                p.applyExternalForce(
                    objectUniqueId=robot_id, linkIndex=-1,
                    forceObj=[1, 0, 0], posObj=[0, 0, 0], flags=p.WORLD_FRAME
                )

            action, _ = controller.model.predict(obs, deterministic=True)
            clipped_action = np.clip(action, env.action_space.low, env.action_space.high)
            if not np.allclose(action, clipped_action):
                print(f"[Diag] Action clipped at step {step}: orig={action}, clipped={clipped_action}")
            action = clipped_action

            obs_next, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Penalty breakdown (update indices if your obs changes)
            pole_angle = obs[2]   # update index if needed!
            base_x = obs[0]
            left_wheel_vel = action[0] if len(action) > 0 else 0.0
            right_wheel_vel = action[1] if len(action) > 1 else 0.0
            angle_penalty = -abs(pole_angle)
            base_penalty = -0.1 * abs(base_x)
            energy_penalty = -0.01 * (left_wheel_vel**2 + right_wheel_vel**2)
            penalty_log["episode"].append(ep)
            penalty_log["step"].append(step)
            penalty_log["angle_penalty"].append(angle_penalty)
            penalty_log["base_penalty"].append(base_penalty)
            penalty_log["energy_penalty"].append(energy_penalty)
            penalty_log["total_reward"].append(reward)

            # --- Terminal logging for each step ---
            print(
                f"[Ep {ep+1:03d} Step {step:03d}] Reward: {reward:+.3f} | "
                f"Angle penalty: {angle_penalty:+.3f}, "
                f"Base penalty: {base_penalty:+.3f}, "
                f"Energy penalty: {energy_penalty:+.3f} | "
                f"Pole angle: {pole_angle:+.3f}, Base x: {base_x:+.3f}, "
                f"Wheel vels: [{left_wheel_vel:+.3f}, {right_wheel_vel:+.3f}]"
            )

            episode_angles.append(pole_angle)
            episode_forces.append(action)
            episode_steps.append(step)
            total_reward += reward

            obs = obs_next

            if done:
                if abs(pole_angle) > 0.8 or step < max_steps - 1:
                    fall_flag = True
                    fall_count += 1
                break

        overshoot = get_overshoot(episode_angles)
        settling_time = get_settling_time(episode_steps, episode_angles)
        smoothness = get_smoothness(episode_angles, episode_steps, dt=1.0)
        energy = get_energy(episode_forces)
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

        print(f"Episode {ep+1}/{episodes} | Reward: {total_reward:.2f}, Overshoot: {overshoot:.3f}, Settling: {settling_time:.2f}, Smoothness: {smoothness:.5f}, Energy: {energy:.2f}, Fall: {fall_rate}")

    # Aggregate metrics
    metrics = ["reward", "overshoot", "settling_time", "smoothness", "energy", "fall_rate", "episode_length"]
    df = pd.DataFrame(log_data)
    df.to_csv("walker_ppo_metrics.csv", index=False)
    print("Saved metrics to walker_ppo_metrics.csv")

    penalty_df = pd.DataFrame(penalty_log)
    penalty_df.to_csv("walker_ppo_penalty_breakdown.csv", index=False)
    print("Saved penalty breakdown to walker_ppo_penalty_breakdown.csv")

    # Penalty means for summary
    summary = {f"{m}_mean": float(df[m].mean()) for m in metrics}
    summary.update({f"{m}_std": float(df[m].std()) for m in metrics})
    summary.update({f"{m}_median": float(df[m].median()) for m in metrics})
    summary["angle_penalty_mean"] = float(penalty_df["angle_penalty"].mean())
    summary["base_penalty_mean"] = float(penalty_df["base_penalty"].mean())
    summary["energy_penalty_mean"] = float(penalty_df["energy_penalty"].mean())

    return df, summary

def log_to_wandb(df, summary):
    episode_table = wandb.Table(dataframe=df)
    wandb.log(summary)
    for metric in df.columns:
        if metric != "episode":
            wandb.log({
                f"{metric}_curve": wandb.plot.line(
                    episode_table, "episode", metric, title=f"{metric.replace('_', ' ').title()} Over Episodes"
                )
            })
    wandb.log({"Evaluation Data": episode_table})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--use_wandb', action='store_true')
    args = parser.parse_args()

    config = load_config(args.config)

    if args.use_wandb:
        wandb.init(
            project="Final_Test_AssistiveWalker",
            config=vars(config),
            name=f"eval_PPO_{datetime.now():%Y%m%d_%H%M%S}",
        )

    env = gym.make("WalkerBalanceContinuousEnv-v0", render_mode="human",test_mode=True)
    env.action_space.seed(getattr(config, "seed", 0))
    np.random.seed(getattr(config, "seed", 0))
    random.seed(getattr(config, "seed", 0))

    model = PPO.load(args.model_path)
    controller = DRLController(model, "PPO", action_space=env.action_space)

    df, summary = evaluate_controller(
        env, controller, getattr(config, "eval_episodes", 100), getattr(config, "max_steps", 1000)
    )

    if args.use_wandb:
        log_to_wandb(df, summary)
        wandb.finish()

    env.close()








