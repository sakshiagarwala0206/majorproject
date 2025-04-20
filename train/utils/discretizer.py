import numpy as np

def create_bins(low, high, bins):
    return [np.linspace(l, h, bins - 1) for l, h in zip(low, high)]

def discretize(obs, bins, obs_low, obs_high):
    obs = np.clip(obs, obs_low, obs_high)
    return tuple(int(np.digitize(x, b)) for x, b in zip(obs, bins))
