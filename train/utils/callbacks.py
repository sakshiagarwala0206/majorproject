from stable_baselines3.common.callbacks import BaseCallback
import wandb
import numpy as np

class CustomCallback(BaseCallback):
    def __init__(self, convergence_threshold=1.0, window_size=50, verbose=0):
        super().__init__(verbose)
        self.rewards = []
        self.convergence_threshold = convergence_threshold
        self.window_size = window_size
        self.convergence_episode = None

    def _on_step(self) -> bool:
        # Log IMU data from the observation (if WandB is initialized)
        obs = self.locals.get("new_obs")
        if obs is not None and len(obs) >= 16:
             if wandb.run is not None:
                wandb.log({
                    "imu_roll": obs[7],
                    "imu_pitch": obs[8],
                    "imu_yaw": obs[9],
                    "imu_ang_vel_x": obs[10],
                    "imu_ang_vel_y": obs[11],
                    "imu_ang_vel_z": obs[12],
                    "imu_lin_acc_x": obs[13],
                    "imu_lin_acc_y": obs[14],
                    "imu_lin_acc_z": obs[15],
                    "global_step": self.num_timesteps
                })
             else:
                # Optionally log a warning or skip logging
                pass



        # Reward and convergence tracking
        if self.locals.get('dones') is not None and any(self.locals['dones']):
            ep_reward = self.locals['infos'][0].get('episode')['r']
            self.rewards.append(ep_reward)

            if len(self.rewards) >= self.window_size:
                avg_now = sum(self.rewards[-self.window_size:]) / self.window_size
                avg_before = sum(self.rewards[-2*self.window_size:-self.window_size]) / self.window_size

                if abs(avg_now - avg_before) < self.convergence_threshold and self.convergence_episode is None:
                    self.convergence_episode = len(self.rewards)
                    print(f"✅ Converged at Episode {self.convergence_episode}!")

        return True


# from stable_baselines3.common.callbacks import BaseCallback

# class CustomCallback(BaseCallback):
#     def __init__(self, convergence_threshold=1.0, window_size=50, verbose=0):
#         super().__init__(verbose)
#         self.rewards = []
#         self.convergence_threshold = convergence_threshold
#         self.window_size = window_size
#         self.convergence_episode = None

#     def _on_step(self) -> bool:
#         if self.locals.get('dones') is not None and any(self.locals['dones']):
#             ep_reward = self.locals['infos'][0].get('episode')['r']
#             self.rewards.append(ep_reward)

#             if len(self.rewards) >= self.window_size:
#                 avg_now = sum(self.rewards[-self.window_size:]) / self.window_size
#                 avg_before = sum(self.rewards[-2*self.window_size:-self.window_size]) / self.window_size

#                 if abs(avg_now - avg_before) < self.convergence_threshold and self.convergence_episode is None:
#                     self.convergence_episode = len(self.rewards)
#                     print(f"✅ Converged at Episode {self.convergence_episode}!")

#         return True
