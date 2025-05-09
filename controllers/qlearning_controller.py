import numpy as np

class QLearningController:
    def __init__(self, q_table):
        self.q_table = q_table

    def act(self, state):
        return int(np.argmax(self.q_table.get(state, [0] * 2)))  # 2 actions for CartPole
