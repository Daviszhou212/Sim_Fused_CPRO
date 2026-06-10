import os
import time
from dataclasses import dataclass


PACKAGE_ROOT = os.path.dirname(os.path.abspath(__file__))
OUTPUT_ROOT = os.path.join(PACKAGE_ROOT, "outputs")
LOG_ROOT = os.path.join(PACKAGE_ROOT, "logs")
TRASH_ROOT = os.path.join(PACKAGE_ROOT, "Trash")


@dataclass
class RunPaths:
    run_id: str
    output_dir: str
    log_dir: str
    trash_dir: str


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


def timestamp_label():
    return time.strftime("%Y%m%d_%H%M%S")


def make_run_paths(run_id, output_root=None, log_root=None, trash_root=None):
    output_base = output_root or OUTPUT_ROOT
    log_base = log_root or LOG_ROOT
    trash_base = trash_root or TRASH_ROOT
    paths = RunPaths(
        run_id=str(run_id),
        output_dir=os.path.abspath(os.path.join(output_base, str(run_id))),
        log_dir=os.path.abspath(os.path.join(log_base, str(run_id))),
        trash_dir=os.path.abspath(os.path.join(trash_base, str(run_id))),
    )
    for path in (paths.output_dir, paths.log_dir, paths.trash_dir):
        ensure_dir(path)
    return paths


def make_compare_run_id(run_tag, episode):
    return "compare_{0}_{1}ep_{2}".format(str(run_tag), int(episode), timestamp_label())

