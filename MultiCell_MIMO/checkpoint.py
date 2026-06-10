from pathlib import Path

import torch


CHECKPOINT_SCHEMA_VERSION = 1


def _to_cpu_state_dict(state_dict):
    cpu_state = {}
    for key, value in dict(state_dict).items():
        if torch.is_tensor(value):
            cpu_state[key] = value.detach().cpu().clone()
        else:
            cpu_state[key] = value
    return cpu_state


def save_checkpoint(checkpoint_root, config, state_dict, stats, episode_index, reason):
    if int(config.get("save_final_checkpoint", 1)) == 0:
        return None

    root = Path(checkpoint_root)
    root.mkdir(parents=True, exist_ok=True)
    filename = "episode_{0:04d}_{1}.pt".format(int(episode_index), str(reason))
    path = root / filename
    payload = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "algorithm": "SLDAC",
        "config": dict(config),
        "state_dict": _to_cpu_state_dict(state_dict),
        "stats": dict(stats),
        "episode_index": int(episode_index),
        "reason": str(reason),
    }
    torch.save(payload, path)
    return str(path)
