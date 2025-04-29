# main.py
import argparse
import yaml
import wandb
from types import SimpleNamespace
import gymnasium
from train.q_learning_trainer import QLearningTrainer

def load_config(path):
    with open(path, 'r') as f:
        cfg = yaml.safe_load(f)
    return SimpleNamespace(**cfg)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args = parser.parse_args()

    config = load_config(args.config)

    wandb.init(project="assistive-walker-qlearning", config=vars(config), save_code=True)
    config = wandb.config

    env = gymnasium.make(config.env)
    trainer = QLearningTrainer(config, env)
    trainer.train()
    wandb.finish()
