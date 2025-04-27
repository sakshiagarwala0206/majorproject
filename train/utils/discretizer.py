import numpy as np

def create_bins(low, high, bins):
    return [np.linspace(l, h, b - 1) for l, h, b in zip(low, high, bins)]

def discretize(obs, bins, obs_low, obs_high):
    obs = np.clip(obs, obs_low, obs_high)
    return tuple(int(np.digitize(x, b)) for x, b in zip(obs, bins))
