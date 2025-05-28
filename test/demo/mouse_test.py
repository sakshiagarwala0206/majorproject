#!/usr/bin/env python
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), './')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

import argparse
import time
import numpy as np
import pybullet as p
import gymnasium as gym
from gymnasium.envs.registration import register
from controllers.drl_controller import DRLController
from stable_baselines3 import PPO

# Register custom env
register(
    id="WalkerBalanceContinuousEnv-v0",
    entry_point="environments.walker:WalkerBalanceContinuousEnv",
    max_episode_steps=500,
)

# Mouse-driven test script

def compute_metrics(angles, torques, timestamps):
    dt = np.diff(timestamps)
    dtheta = np.diff(angles)
    smoothness = float(np.mean(np.abs(dtheta / dt))) if len(dt) > 0 else 0.0
    overshoot = float(np.max(np.abs(angles)))
    tol = 0.05
    settled = np.where(np.abs(angles) < tol)[0]
    settling_time = float(timestamps[settled[0]]) if settled.size > 0 else float(timestamps[-1])
    energy = float(np.sum(np.abs(torques)))
    return overshoot, settling_time, smoothness, energy


def main(model_path):
    print("\n=== Starting mouse-driven test ===\n")
    # Load model and env
    model = PPO.load(model_path)
    env = gym.make("WalkerBalanceContinuousEnv-v0", render_mode="human")
    base_env = env.unwrapped
    dt = base_env.time_step
    controller = DRLController(model, "PPO", env.action_space)

    try:
        while True:
            obs, _ = env.reset()
            print("Environment reset. Waiting for click disturbance...")
            # Wait for click
            while True:
                env.render()
                # small sleep to avoid tight loop
                time.sleep(0.001)
                # detect if disturbance applied by checking joint torque
                js = p.getJointState(base_env.robot_id, base_env.pole_joint_index)
                if abs(js[3]) > 1e-4:
                    print("Disturbance applied!")
                    break

            # Record recovery
            angles, torques, times = [], [], []
            t0 = time.time()
            consec = 0
            while True:
                action, _ = controller.model.predict(obs, deterministic=True)
                obs, reward, done, _, _ = env.step(action)
                p.stepSimulation()
                now = time.time() - t0
                angles.append(obs[0])
                torques.append(np.linalg.norm(action))
                times.append(now)

                if abs(obs[0]) < 0.05:
                    consec += 1
                else:
                    consec = 0
                if consec >= 30 or now > 5.0 or done:
                    break
                time.sleep(dt)

            over, sett, smooth, energy = compute_metrics(angles, torques, times)
            print(f"Recovery metrics: Overshoot={over:.3f}, Settling={sett:.3f}s, Smoothness={smooth:.3f}, Energy={energy:.3f}\n")

    except KeyboardInterrupt:
        print("\nInterrupted by user, exiting.")
    finally:
        env.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()
    main(args.model_path)
