import os
import sys
import gymnasium as gym
import numpy as np
import wandb
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from wandb.integration.sb3 import WandbCallback
from datetime import datetime
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from train.utils.logger import setup_logger
from train.utils.config_loader import load_config

logger = setup_logger()

class BaseTrainer:
    def __init__(self, algo_name: str, config: dict, env_id: str):
        self.algo_name = algo_name
        self.config = config
        self.env_id = env_id

        self._init_wandb()
        self.env = self._init_env()
        self.action_noise = self._init_noise()

    def _init_wandb(self):
        wandb.init(
            project="assistive-walker-drl",
            config=self.config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
        )
        self.wandb_config = wandb.config

    def _init_env(self):
        env = gym.make(self.env_id, render_mode=None)
        env = Monitor(env)
        return env

    def _init_noise(self):
        """Initialize action noise if the action space is continuous (Box)."""
        import numpy as np
        import gymnasium as gym  # <<<<<<<< CAREFUL: use gymnasium, not gym
        
        print(f"DEBUG: action_space = {self.env.action_space}")
        print(f"DEBUG: type(action_space) = {type(self.env.action_space)}")
        
        if isinstance(self.env.action_space, gym.spaces.Discrete):
            print("DEBUG: Discrete space detected. No action noise needed.")
            return None

        print("DEBUG: Continuous space detected.")

        action_noise_std = getattr(self.wandb_config, 'action_noise', 0.1)

        n_actions = self.env.action_space.shape[0]

        if action_noise_std == 0:
            return None

        return NormalActionNoise(
            mean=np.zeros(n_actions),
            sigma=action_noise_std * np.ones(n_actions)
        )

    def get_callbacks(self):
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=f"./models/{self.algo_name.lower()}/",
            name_prefix=f"{self.algo_name.lower()}_cartpole",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"./models/{self.algo_name.lower()}/",
            verbose=2,
        )
        return [checkpoint_callback, wandb_callback]
    
    

    def generate_experiment_id(controller_name, config_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{controller_name}_{config_name}_{timestamp}"


    def finish(self):
        wandb.finish()
