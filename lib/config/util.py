"""config utilities."""
lib_name = "lib"

import os

def get_config_dir():
    """Get the absolute path to the config file.

    Returns
    -------
    dirname : str
        Path to the config directory
    """

    default_dir = os.path.join(os.getcwd(),lib_name, "config","config_data")

    return default_dir