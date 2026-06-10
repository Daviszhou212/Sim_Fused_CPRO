from pathlib import Path


def ensure_dir(path):
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def build_output_path(output_root, algorithm, filename, allow_overwrite=False):
    root = ensure_dir(Path(output_root) / str(algorithm))
    path = root / str(filename)
    if path.exists() and not bool(allow_overwrite):
        raise FileExistsError("refusing to overwrite existing output: {0}".format(path))
    return path


def build_checkpoint_dir(checkpoint_root, algorithm, run_tag, seed, run_id=None):
    root = Path(checkpoint_root) / str(algorithm) / str(run_tag)
    if run_id:
        root = root / str(run_id)
    return ensure_dir(root / "seed_{0}".format(int(seed)))


def build_trash_dir(base_dir="MultiCell_MIMO"):
    return ensure_dir(Path(base_dir) / "Trash")
