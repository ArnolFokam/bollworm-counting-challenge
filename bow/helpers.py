import os
import json
import random
import string

import torch
import numpy as np

def seed_everything(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def get_dir(*paths) -> str:
    """Creates a dir from a list of directories (like os.path.join), runs os.makedirs and returns the name
    Returns:
        str:
    """
    directory = os.path.join(*paths)
    os.makedirs(directory, exist_ok=True)
    return directory
    
def generate_random_string(length: int = 10):
    return ''.join(random.choices(string.ascii_uppercase + string.digits, k=length))

def reset_wandb_env():
    exclude = {
        "WANDB_PROJECT",
        "WANDB_ENTITY",
        "WANDB_API_KEY",
    }
    for k, v in os.environ.items():
        if k.startswith("WANDB_") and k not in exclude:
            del os.environ[k]