import pickle
import numpy as np
import wandb
from utils.evaluate import run_episode
from config.qlearning_config import Q_TABLE_PATH, TEST_EPISODES

class QLearningController:
    def __init__(self, q_table_path):
        with open(q_table_path, "rb") as f:
            self.q_table, self.bins = pickle.load(f)

    def discretize(self, obs):
        return tuple(np.digitize(o, b) for o, b in zip(obs, self.bins))

    def act(self, obs):
        state = self.discretize(obs)
        return np.argmax(self.q_table[state])

def evaluate_qlearning():
    # Initialize controller
    controller = QLearningController(Q_TABLE_PATH)

    # Initialize environment (replace with your actual environment)
    env = make_env(seed=42)

    # Initialize WandB logging
    wandb.init(project="q_learning_evaluation")

    # Collect results
    results = []
    for ep in range(TEST_EPISODES):
        metrics = run_episode(env, controller, max_steps=500)
        metrics["episode"] = ep
        wandb.log(metrics)  # Log to WandB
        results.append(metrics)

    # Save results to CSV
    df = pd.DataFrame(results)
    df.to_csv("qlearning_results.csv", index=False)

    wandb.finish()
