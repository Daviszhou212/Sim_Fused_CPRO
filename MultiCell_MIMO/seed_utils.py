import random

import numpy as np
import torch


def set_global_seed(seed):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    return seed


def resolve_torch_device(device):
    text = "cpu" if device is None else str(device).strip().lower()
    if text in ("", "auto"):
        text = "cuda" if torch.cuda.is_available() else "cpu"
    if text.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(text)
