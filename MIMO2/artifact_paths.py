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


def _format_seed_dir(seed):
    return "seed_{0}".format(int(seed))


def _normalize_compare_run_tag(run_tag):
    text = "" if run_tag is None else str(run_tag).strip()
    return text


def get_compare_output_dir(base_dir, create=True, seed=None, run_tag=None):
    compare_dir = os.path.join(get_outputs_root(base_dir, create=create), COMPARE_DIRNAME)
    compare_dir = _ensure_dir(compare_dir, create)
    if seed is None:
        target_dir = compare_dir
    else:
        target_dir = os.path.join(compare_dir, _format_seed_dir(seed))
        target_dir = _ensure_dir(target_dir, create)
    normalized_run_tag = _normalize_compare_run_tag(run_tag)
    if not normalized_run_tag:
        return target_dir
    run_tag_dir = os.path.join(target_dir, normalized_run_tag)
    return _ensure_dir(run_tag_dir, create)


def build_algorithm_artifact_path(base_dir, algorithm_name, filename, create=True):
    return os.path.join(get_algorithm_output_dir(base_dir, algorithm_name, create=create), str(filename))


def build_compare_artifact_path(base_dir, filename, create=True, seed=None, run_tag=None):
    return os.path.join(
        get_compare_output_dir(base_dir, create=create, seed=seed, run_tag=run_tag),
        str(filename),
    )


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
