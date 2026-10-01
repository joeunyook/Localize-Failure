import json
from pathlib import Path

import numpy as np


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def save_residuals(tensor, path):
    """tensor: torch (n_layers+1, d_model) float16 -> .npy"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, tensor.numpy())
