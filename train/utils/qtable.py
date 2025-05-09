import numpy as np

def convert_q_table_to_dict(q_table, action_space):
    """
    Convert a NumPy Q-table to a dictionary format.
    Args:
        q_table (np.ndarray): The trained Q-table (NumPy array).
        action_space (int): The number of possible actions in the environment.
    
    Returns:
        dict: The Q-table as a dictionary with state-action pairs.
    """
    q_table_dict = {}
    it = np.nditer(q_table, flags=['multi_index'])
    for val in it:
        idx = it.multi_index[:-1]  # state indices
        action = it.multi_index[-1]  # action index
        if idx not in q_table_dict:
            q_table_dict[idx] = [0] * action_space
        q_table_dict[idx][action] = float(val)
    return q_table_dict
