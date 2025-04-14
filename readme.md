
# 🧠 DDPG Inference on Custom Inverted Pendulum (PyBullet + Gymnasium)

This project demonstrates real-time **inference** using a pre-trained **DDPG (Deep Deterministic Policy Gradient)** model on a **custom inverted pendulum environment** built using **PyBullet** and **Gymnasium**.

---

## 📦 Project Structure

```
.
├── cartpole_env.py                 # Custom environment (PyBullet + Gymnasium API)
├── inference_ddpg_pendulum.py     # Inference script with action noise
├── ddpg_inverted_pendulum.zip     # Trained DDPG model (loaded via SB3)
├── pendulum.urdf                  # URDF file describing the robot
├── README.md                      # This file
```

---

## ⚙️ Environment Overview

Your environment simulates an **inverted pendulum system** with:

- **Slider joint** for cart movement (prismatic joint).
- **Hinge joint** for the pole (revolute joint).
- Uses **PyBullet** for physics simulation.
- Complies with **Gymnasium**'s API (`reset`, `step`, `render`, etc.).

> Ensure your `InvertedPendulumEnvGymnasium` class is correctly registered or imported in `cartpole_env.py`.

---

## 🧪 Inference Script Breakdown

File: `inference_ddpg_pendulum.py`

### 🔹 Key Features

- Loads a pre-trained DDPG model from Stable Baselines3.
- Adds **Gaussian noise** to actions during inference to encourage slight exploration.
- Uses `env.step()` to interact with the custom environment.
- Tracks **pole angle** to detect "balanced" states.
- Stops an episode early if the pole stays balanced for 50 steps.

### 🔸 Action Noise

```python
from stable_baselines3.common.noise import NormalActionNoise
action_noise = NormalActionNoise(mean=np.zeros(env.action_space.shape),
                                  sigma=0.05 * np.ones(env.action_space.shape))
```

This simulates small deviations from deterministic policy actions for robustness.

---

## ▶️ Running Inference

Make sure all dependencies are installed:

```bash
pip install stable-baselines3 gymnasium[all] pybullet numpy
```

Then run:

```bash
python inference_ddpg_pendulum.py
```

### Expected Behavior

- A window opens showing the pendulum balancing in real time.
- The script prints rewards, actions, and pole angle.
- If the pole stays within ±0.05 radians for 50 consecutive steps, the episode ends early.

---

## 📊 Observations and Debugging

- `obs[2]` is assumed to be the **pole angle**.
- Adjust noise levels or pole angle thresholds as needed.
- For persistent debugging/logging, consider logging:
  - Actions
  - Rewards
  - Pole angle
  - Episode length

---

## 🧠 DDPG Notes

- Suitable for continuous action spaces like motor torques.
- Works well with low-dimensional state vectors like inverted pendulum angles and velocities.

---

## 🚀 Next Steps

- 🎥 Add video recording using `gymnasium.wrappers.RecordVideo`
- 📉 Plot reward curve over episodes
- 🛠️ Test different levels of noise
- 📦 Run inference on a Raspberry Pi or embedded device
- 🧪 Compare with PPO or SAC

---

Made with ❤️ for balancing robots.

```

