import yaml

def load_config(config_path: str) -> dict:
    """
    Load a YAML configuration file.

    Args:
        config_path (str): Path to the YAML config file.

    Returns:
        dict: Loaded configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config
