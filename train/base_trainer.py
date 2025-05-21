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
from utils.callbacks import CustomCallback 
from stable_baselines3.common.callbacks import BaseCallback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))
import numpy as np
from utils.logger import setup_logger
from utils.config_loader import load_config

logger = setup_logger()
class StopTrainingOnPatience(BaseCallback):
    def __init__(self, min_improvement: float = 1.0, patience: int = 10):
        super().__init__()
        self.min_improvement = min_improvement
        self.patience = patience
        self.patience_counter = 0
        self.last_reward = -float('inf')

    def _on_step(self) -> bool:
        # Calculate the current reward improvement
        current_reward = self.locals["rewards"][-1]  # Get the last reward
        reward_improvement = abs(self.last_reward - current_reward)

        # Check if the improvement is less than the minimum improvement
        if reward_improvement < self.min_improvement:
            self.patience_counter += 1
        else:
            self.patience_counter = 0

        self.last_reward = current_reward

        # Stop training if patience limit is reached
        if self.patience_counter >= self.patience:
            print(f"Training stopped due to no significant improvement for {self.patience} episodes.")
            return False  # This will stop training

        return True  # Continue training

class BaseTrainer:
    def __init__(self, algo_name: str, config: dict, env_id: str,run_name=None):
        self.algo_name = algo_name
        self.config = config
        self.env_id = env_id

        
        if run_name is None:
            self.run_name = f"train_{self.algo_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            self.run_name = run_name

        self._init_wandb()
        self.env = self._init_env()
        self.action_noise = self._init_noise()
            
    def _init_wandb(self):
        wandb.init(
            project="cartpole-DRL-train",
            config=self.config,
            sync_tensorboard=True,
            monitor_gym=True,
            save_code=True,
            name=self.run_name,
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
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        name_prefix = f"{self.algo_name.lower()}_{timestamp}"
        checkpoint_callback = CheckpointCallback(
            save_freq=10_000,
            save_path=f"./models/{self.algo_name.lower()}/{timestamp}/",
            name_prefix=f"{self.algo_name.lower()}_walker",
            save_replay_buffer=True,
            save_vecnormalize=True,
        )
        wandb_callback = WandbCallback(
            gradient_save_freq=100,
            model_save_path=f"./models/{self.algo_name.lower()}/{timestamp}/",
            verbose=2,
        )

        custom_callback = CustomCallback(
        convergence_threshold=0.5,
        window_size=20,
        )
        # Add the new StopTrainingOnRewardThreshold callback here
        # Add the StopTrainingOnPatience callback here
        # Here you can return your custom StopTrainingOnPatience callback along with other callbacks
        
        patience_callback = StopTrainingOnPatience(min_improvement=1.0, patience=200000)
        self.custom_callback = custom_callback
        return [checkpoint_callback, wandb_callback,custom_callback, patience_callback]

    

    def generate_experiment_id(controller_name, config_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M")
        return f"{controller_name}_{config_name}_{timestamp}"


    def finish(self):
        wandb.finish()
