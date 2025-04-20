import pickle
import logging

logger = logging.getLogger(__name__)

def save_q_table(q_table, episode, path="./checkpoints"):
    filename = f"{path}/q_table_ep{episode}.pkl"
    with open(filename, "wb") as f:
        pickle.dump(q_table, f)
    logger.info(f"✅ Q-table saved at episode {episode}")

def load_q_table(path):
    with open(path, "rb") as f:
        q_table = pickle.load(f)
    logger.info(f"✅ Loaded Q-table from {path}")
    return q_table

def save_metadata(metadata, filename="training_metadata_001.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(metadata, f)
    logger.info("✅ Metadata saved.")
