from pathlib import Path

import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1


def _to_checkpoint_value(value):
    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _to_checkpoint_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_checkpoint_value(item) for item in value]
    return value


def save_checkpoint(checkpoint_root, config, state_dict, stats, episode_index, reason):
    if int(config.get("save_final_checkpoint", 1)) == 0:
        return None

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    filename = "episode_{0:04d}_{1}.pt".format(int(episode_index), str(reason))
    path = root / filename
    if path.exists() and not bool(int(config.get("allow_overwrite", 0))):
        raise FileExistsError("refusing to overwrite existing checkpoint: {0}".format(path))
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "SLDAC",
        "config": _to_checkpoint_value(dict(config)),
        "state_dict": _to_checkpoint_value(dict(state_dict)),
        "stats": _to_checkpoint_value(dict(stats)),
        "episode_index": int(episode_index),
        "reason": str(reason),
    }
    torch.save(payload, path)
    return str(path)
