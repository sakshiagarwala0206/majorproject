#!/bin/bash

echo "Training DDPG..."
python train/train_ddpg.py

echo "Training DQN..."
python train/train_dqn.py

echo "Training PPO..."
python train/train_ppo.py

echo "Training SAC..."
python train/train_sac.py
