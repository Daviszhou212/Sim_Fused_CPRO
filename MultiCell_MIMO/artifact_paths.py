from pathlib import Path


def ensure_dir(path):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def build_output_path(output_root, algorithm, filename):
    root = ensure_dir(Path(output_root) / str(algorithm))
    return root / str(filename)


def build_checkpoint_dir(checkpoint_root, algorithm, run_tag, seed):
    return ensure_dir(Path(checkpoint_root) / str(algorithm) / str(run_tag) / "seed_{0}".format(int(seed)))


def build_trash_dir(base_dir="MultiCell_MIMO"):
    return ensure_dir(Path(base_dir) / "Trash")
