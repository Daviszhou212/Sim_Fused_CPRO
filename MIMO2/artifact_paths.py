import os


OUTPUTS_DIRNAME = "outputs"
COMPARE_DIRNAME = "compare"


def _normalize_base_dir(base_dir):
    return os.path.abspath(str(base_dir))


def _ensure_dir(path, create):
    if create:
        os.makedirs(path, exist_ok=True)
    return path


def get_outputs_root(base_dir, create=True):
    root = os.path.join(_normalize_base_dir(base_dir), OUTPUTS_DIRNAME)
    return _ensure_dir(root, create)


def get_algorithm_output_dir(base_dir, algorithm_name, create=True):
    output_dir = os.path.join(get_outputs_root(base_dir, create=create), str(algorithm_name))
    return _ensure_dir(output_dir, create)


def get_compare_output_dir(base_dir, create=True):
    compare_dir = os.path.join(get_outputs_root(base_dir, create=create), COMPARE_DIRNAME)
    return _ensure_dir(compare_dir, create)


def build_algorithm_artifact_path(base_dir, algorithm_name, filename, create=True):
    return os.path.join(get_algorithm_output_dir(base_dir, algorithm_name, create=create), str(filename))


def build_compare_artifact_path(base_dir, filename, create=True):
    return os.path.join(get_compare_output_dir(base_dir, create=create), str(filename))


def resolve_algorithm_artifact_path(base_dir, algorithm_name, filename):
    new_path = build_algorithm_artifact_path(base_dir, algorithm_name, filename, create=False)
    if os.path.exists(new_path):
        return new_path

    legacy_path = os.path.join(_normalize_base_dir(base_dir), str(filename))
    if os.path.exists(legacy_path):
        return legacy_path

    raise FileNotFoundError(
        "artifact not found in outputs/{0} or legacy root: {1}".format(
            algorithm_name,
            filename,
        )
    )
