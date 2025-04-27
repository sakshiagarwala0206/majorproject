import numpy as np
import pickle

class RLAgent:
    def __init__(self, q_table=None, bins=None, obs_low=None, obs_high=None, model_path=None):
        # If model_path is provided, load the model
        if model_path:
            with open(model_path, "rb") as f:
                self.q_table, self.bins = pickle.load(f)
        elif q_table is not None and bins is not None:
            # If q_table and bins are provided, use them
            self.q_table = q_table
            self.bins = bins
        else:
            raise ValueError("Either model_path or q_table and bins must be provided.")
        
        self.obs_low = obs_low
        self.obs_high = obs_high

    def discretize(self, obs):
        return tuple(np.digitize(obs[i], self.bins[i]) for i in range(len(obs)))

    def act(self, obs):
        state = self.discretize(obs)
        # Ensure state is within valid bounds for Q-table
        if any(s >= len(b) for s, b in zip(state, self.bins)):
                print(f"Invalid state: {state}")
                return 0  # Default action if invalid state
        return int(np.argmax(self.q_table[state]))

    def save_model(self, path):
        with open(path, "wb") as f:
            pickle.dump((self.q_table, self.bins), f)