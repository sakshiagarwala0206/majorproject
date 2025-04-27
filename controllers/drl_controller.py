import torch
import torch.nn.functional as F
import numpy as np

class DRLController:
    def __init__(self, model, model_type, action_space):
        """
        Initialize the DRL Controller.

        :param model: The RL model (DQN, DDPG, SAC, PPO, etc.)
        :param model_type: The type of model (e.g., 'dqn', 'ddpg', 'sac', 'ppo')
        :param action_space: The action space of the environment (discrete or continuous)
        """
        self.model = model
        self.model_type = model_type
        self.action_space = action_space
    def act(self, observation):
        """
        Select an action based on the given observation.

        :param observation: The state of the environment
        :return: Action selected by the controller
        """
        # Convert observation into tensor format for model input
        observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

        with torch.no_grad():  # No need to compute gradients during evaluation
            if self.model_type.lower() == 'dqn':  # Discrete action space
                # For DQN, use the model's predict method (not the model itself)
                q_values, _ = self.model.predict(observation_tensor, deterministic=True)
                action = q_values  # This should be a single action, as DQN predicts a discrete action

            elif self.model_type == 'ddpg' or self.model_type == 'sac':  # Continuous action space
                # For DDPG/SAC, directly predict actions (continuous)
                action = self.model(observation_tensor)
                action = action.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

                # Ensure the action is within bounds (optional but recommended for continuous spaces)
                if isinstance(self.action_space, np.ndarray):
                    action = np.clip(action, self.action_space.low, self.action_space.high)

            elif self.model_type == 'ppo':  # Discrete or continuous action space
                # For PPO, select action from a distribution
                action_distribution = self.model(observation_tensor)
                action = action_distribution.sample().cpu().numpy()

            else:
                raise NotImplementedError(f"Model type {self.model_type} not supported for action selection")

        return action

    # def act(self, observation):
    #     """
    #     Select an action based on the given observation.

    #     :param observation: The state of the environment
    #     :return: Action selected by the controller
    #     """
    #     # Convert observation into tensor format for model input
    #     observation_tensor = torch.tensor(observation, dtype=torch.float32).unsqueeze(0)  # Add batch dimension

    #     with torch.no_grad():  # No need to compute gradients during evaluation
    #         if self.model_type.lower() == 'dqn':  # Discrete action space
    #             # For DQN, select action with the highest Q-value
    #             q_values = self.model(observation_tensor)
    #             action = torch.argmax(q_values, dim=1).item()

    #         elif self.model_type == 'ddpg' or self.model_type == 'sac':  # Continuous action space
    #             # For DDPG/SAC, we directly predict actions (continuous)
    #             action = self.model(observation_tensor)
    #             action = action.squeeze(0).cpu().numpy()  # Remove batch dimension and convert to numpy

    #             # Ensure the action is within bounds (optional but recommended for continuous spaces)
    #             if isinstance(self.action_space, np.ndarray):
    #                 action = np.clip(action, self.action_space.low, self.action_space.high)

    #         elif self.model_type == 'ppo':  # Discrete or continuous action space
    #             # For PPO, select action from a distribution
    #             action_distribution = self.model(observation_tensor)
    #             action = action_distribution.sample().cpu().numpy()

    #         else:
    #             raise NotImplementedError(f"Model type {self.model_type} not supported for action selection")

    #     return action
