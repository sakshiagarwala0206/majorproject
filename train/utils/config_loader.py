import argparse
import yaml

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config file')

    # Add optional overrides
    parser.add_argument('--run_name', type=str)
    parser.add_argument('--use_gui', action='store_true')
    parser.add_argument('--test_mode', action='store_true')
    # Add more as needed...

    args, _ = parser.parse_known_args()
    config = load_config(args.config)

    # Merge CLI overrides into config
    for key, value in vars(args).items():
        if value is not None:
            config[key] = value

    return config
#         policy_kwargs=dict(

# import yaml

# def load_config(config_path: str) -> dict:
#     """
#     Load a YAML configuration file.

#     Args:
#         config_path (str): Path to the YAML config file.

#     Returns:
#         dict: Loaded configuration as a dictionary.
#     """
#     with open(config_path, 'r') as f:
#         config = yaml.safe_load(f)
#     return config
