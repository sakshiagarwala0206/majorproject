```markdown
# Comparative Analysis of Deep Reinforcement Learning, Traditional RL, and PID Control for Assistive Walker and CartPole Systems

## Overview

This repository provides a unified, research-focused platform for benchmarking classical and modern control strategies-including PID, Q-Learning, PPO, SAC, and DDPG-on two custom robotic systems: an **Assistive Walker** and a **CartPole**. Both systems are modeled using URDF for realistic physics and simulated in PyBullet, with custom Gymnasium environments for reinforcement learning research[1][2][3].

---

## Table of Contents

- [Project Objectives](#project-objectives)
- [System Architecture](#system-architecture)
- [Assistive Walker](#assistive-walker)
- [CartPole](#cartpole)
- [Custom Environment Creation](#custom-environment-creation)
- [Control Algorithms](#control-algorithms)
- [Training & Evaluation Pipeline](#training--evaluation-pipeline)
- [How to Run](#how-to-run)
- [File Structure](#file-structure)
- [Research Insights](#research-insights)
- [References](#references)
- [License](#license)

---

## Project Objectives

- Develop custom URDF models for both systems, capturing realistic mechanical properties[1][2][3].
- Implement Gymnasium-compatible environments using PyBullet for physics simulation.
- Train and benchmark PID, Q-Learning, PPO, SAC, and DDPG controllers.
- Compare performance using metrics such as episode length, cumulative reward, and stability.
- Analyze strengths and limitations of each control strategy for both robots.

---

## System Architecture

| Layer         | Description                                                      |
|---------------|------------------------------------------------------------------|
| URDF Model    | Defines robot structure, joints, inertia, friction, and sensors. |
| PyBullet      | Loads URDF, simulates physics, provides state and control APIs.  |
| Environment   | Gymnasium-compatible class: defines observations, actions, rewards, and episode logic. |
| RL Algorithm  | Agent interacts with environment, learns to optimize reward.     |

---

## Assistive Walker

**Description:**  
A differential-drive walker with two powered wheels, an assistive handle (pole), and a simulated IMU sensor. Designed for research in stabilization, navigation, and user-adaptive control[1][3].

**URDF Highlights:**
- **Base:** Rigid box, 4.0 kg, realistic inertia.
- **Wheels:** Two, each 0.8 kg, high friction for realistic drive.
- **Pole:** 1.2 kg, 1.0 m, revolute joint for handle dynamics.
- **IMU:** Simulated MPU6050, provides orientation, angular velocity, and linear acceleration data[1][3].

**Environment:**
- **Observation:** Pole angle/velocity, base pose, wheel velocities, IMU data.
- **Action:**
  - *Discrete:* {left, right, stop}
  - *Continuous:* [left_wheel_torque, right_wheel_torque]
- **Reward:** Penalizes pole deviation, displacement, and excessive wheel velocity.
- **Termination:** Pole falls or walker moves out of bounds[1][3].

---

## CartPole

**Description:**  
A classic inverted pendulum system with a sliding cart and a pole, implemented with a custom URDF for realistic simulation[2].

**URDF Highlights:**
- **Track:** Fixed, 30x0.05x0.05 m, visual only.
- **Cart:** 0.5x0.5x0.2 m, 4 kg, prismatic joint for horizontal motion.
- **Pole:** 1.0 m, 1 kg, continuous joint for rotation.
- **Friction/Damping:** Realistic values for both cart and pole to ensure stable, physical behavior[2].

**Environment:**
- **Observation:** Cart position/velocity, pole angle/velocity.
- **Action:**
  - *Discrete:* {left, right}
  - *Continuous:* Apply force/torque to cart.
- **Reward:** +1 per step pole remains balanced.
- **Termination:** Pole falls or cart moves off track[2].

---

## Custom Environment Creation

Both environments are implemented as Python classes inheriting from `gymnasium.Env`.

**Key Steps:**
1. **URDF Modeling:** Define robot structure and joints.
2. **PyBullet Integration:** Load URDF, set up physics.
3. **Observation & Action Spaces:** Define what the agent sees and controls.
4. **Reward & Episode Logic:** Specify how agents are scored and when episodes end.
5. **Registration:** Register with Gymnasium for use in RL pipelines[1][2][3].

**Example Usage:**
```
# Assistive Walker (Continuous)
from environments.walker import AssistiveWalkerContinuousEnv
env = AssistiveWalkerContinuousEnv()
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
env.close()

# CartPole (Continuous)
from environments.cartpole import CartPoleContinuousEnv
env = CartPoleContinuousEnv()
obs = env.reset()
done = False
while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)
env.close()
```

**How URDF, PyBullet, Environment, and RL Connect:**
- **URDF:** Blueprint for robot/system.
- **PyBullet:** Loads URDF, simulates physics.
- **Environment:** Defines RL interface (observations, actions, rewards).
- **RL Agent:** Interacts with environment, learns control policy[2].

---

## Control Algorithms

| Algorithm   | Type         | Action Space   | Library            | Notes                                     |
|-------------|--------------|---------------|--------------------|-------------------------------------------|
| PID         | Classical    | Continuous    | Custom             | Baseline for comparison                   |
| Q-Learning  | RL (classic) | Discrete      | Stable Baselines3  | Value-based, tabular                      |
| PPO         | Deep RL      | Continuous    | Stable Baselines3  | On-policy, robust, stable                 |
| SAC         | Deep RL      | Continuous    | Stable Baselines3  | Off-policy, sample-efficient              |
| DDPG        | Deep RL      | Continuous    | Stable Baselines3  | Off-policy, deterministic                 |

All RL algorithms are trained and evaluated using Stable Baselines3, with custom wrappers for noise and logging[1][2][3].

---

## Training & Evaluation Pipeline

1. **Configure Environment:** Choose robot and action space.
2. **Select Algorithm:** PID, Q-Learning, PPO, SAC, or DDPG.
3. **Train:** Run training loop with chosen algorithm and hyperparameters.
4. **Evaluate:** Test trained policy, collect metrics (reward, episode length, stability).
5. **Analyze:** Compare across algorithms and environments for research insights[1][2][3].

**Config Example (PPO):**
```
policy: MlpPolicy
learning_rate: 0.0003
gamma: 0.99
batch_size: 64
n_steps: 2048
total_timesteps: 1000000
action_noise: 0.1
wandb_project: assistive-walker-ppo
```

---

## How to Run

1. **Install Dependencies**
   ```
   pip install gymnasium pybullet stable-baselines3 wandb
   ```

2. **Train PPO on Assistive Walker**
   ```
   python train/ppo_trainer.py --config configs/ppo_config.yaml
   ```

3. **Train PPO on CartPole**
   ```
   python train/ppo_trainer.py --config configs/cartpole_ppo_config.yaml
   ```

4. **Monitor Training**
   - Use Weights & Biases for logging and visualization.

---

## File Structure

```
urdf/
  walker.urdf           # Assistive Walker URDF
  cartpole.urdf         # CartPole URDF
environments/
  walker.py             # Assistive Walker environment
  cartpole.py           # CartPole environment
train/
  basetrainer.py        # Base trainer class
  ppo_trainer.py        # PPO training script
  utils/
    callbacks.py        # Custom callbacks
    logger.py           # Logging utilities
    configloader.py     # Config loader
configs/
  ppo_config.yaml           # PPO config for walker
  cartpole_ppo_config.yaml  # PPO config for cartpole
README.md               # This file
```

---

## Research Insights

- **PID:** Fast, interpretable, but limited adaptability to nonlinearities and disturbances.
- **Q-Learning:** Effective for simple, discrete tasks, but does not scale well to high-dimensional or continuous domains.
- **Deep RL (PPO, SAC, DDPG):** Superior performance in complex, noisy, and continuous environments; robust to varied initial conditions.
- **IMU Integration (Walker):** Enhances state estimation and reward shaping for robust learning.
- **Realistic Physics (URDF + PyBullet):** Ensures that learned policies are physically plausible and transferable[1][2][3].

---

## References

- See `Assistive-Walker-Documentation.pdf`, `Custom-assistive-walker-code-used.pdf`, and `Cart-Pole-Documentation.pdf` for full technical and code details[1][2][3].

---

## License

This project is licensed under the MIT License.

---

**Contact:**  
For technical questions or collaboration, open an issue or contact the maintainers.

---
```
