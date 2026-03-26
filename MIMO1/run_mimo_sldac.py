import argparse
import os
import shutil

import numpy as np
from scipy.io import savemat

from artifact_paths import build_algorithm_artifact_path
from seed_utils import apply_python_config_priority, format_ignored_cli_overrides, resolve_experiment_seeds
from SLDAC import SLDAC_main


# Keep only the four SLDAC settings needed for current reproduction.
# Tuple fields: output suffix, log label, T, grad_T, num_new_data, Q_update_time.
SLDAC_RUNS = [
    ("b100_q1", "SLDAC, batchsize=100, q=1", 500, 500, 100, 1),
    ("b100_q5", "SLDAC, batchsize=100, q=5", 500, 500, 100, 5),
    ("b100_q10", "SLDAC, batchsize=100, q=10", 500, 500, 100, 10),
    ("b500_q10", "SLDAC, T=500, batchsize=500, q=10", 50, 100, 100, 10),
]

DEFAULT_SEED = 1
DEFAULT_SEEDS = (5,6,7,8,9)
ALGORITHM_NAME = "SLDAC"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LEGACY_CHECKPOINT_PREFIX = "episode_"
LEGACY_CHECKPOINT_SUFFIX = ".pt"


def _format_seed_dir(seed):
    return "seed_{0}".format(int(seed))


def _format_seed_list_text(seed_values):
    return ",".join(str(int(seed_value)) for seed_value in seed_values)


def _resolve_checkpoint_root(checkpoint_root):
    root = checkpoint_root
    if not root:
        root = os.path.join("checkpoints", "SLDAC")
    root = str(root)
    if not os.path.isabs(root):
        root = os.path.join(os.getcwd(), root)
    return root


def _is_legacy_checkpoint_file(filename):
    text = str(filename)
    return text.startswith(LEGACY_CHECKPOINT_PREFIX) and text.endswith(LEGACY_CHECKPOINT_SUFFIX)


def _migrate_legacy_checkpoints(checkpoint_root, example_name, default_seed=DEFAULT_SEED):
    example_dir = os.path.join(_resolve_checkpoint_root(checkpoint_root), str(example_name))
    if not os.path.isdir(example_dir):
        return

    for run_tag in os.listdir(example_dir):
        run_dir = os.path.join(example_dir, run_tag)
        if not os.path.isdir(run_dir):
            continue

        legacy_filenames = []
        for name in os.listdir(run_dir):
            path = os.path.join(run_dir, name)
            if os.path.isfile(path) and _is_legacy_checkpoint_file(name):
                legacy_filenames.append(name)
        if not legacy_filenames:
            continue

        seed_dir = os.path.join(run_dir, _format_seed_dir(default_seed))
        os.makedirs(seed_dir, exist_ok=True)
        for filename in legacy_filenames:
            source_path = os.path.join(run_dir, filename)
            target_path = os.path.join(seed_dir, filename)
            if os.path.exists(target_path):
                continue
            shutil.move(source_path, target_path)
            print("migrated legacy checkpoint:", source_path, "->", target_path)


def _build_mat_metadata(args, algorithm, run_tag):
    return {
        "seed": np.asarray([[int(getattr(args, "seed", DEFAULT_SEED))]], dtype=np.int32),
        "algorithm": np.asarray([str(algorithm)], dtype="U32"),
        "run_tag": np.asarray([str(run_tag)], dtype="U32"),
    }


def _save_mat_with_seed(filename, payload, args, algorithm, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, algorithm, run_tag))
    root, ext = os.path.splitext(str(filename))
    seed_value = int(getattr(args, "seed", DEFAULT_SEED))
    savemat("{0}_seed{1}{2}".format(root, seed_value, ext), full_payload)


def _run_single_seed(args, example_name):
    for output_suffix, message, t_horizon, grad_t, num_new_data_run, q_update_time in SLDAC_RUNS:
        print(message)
        args.run_tag = output_suffix
        args.T = t_horizon
        args.grad_T = grad_t
        args.num_new_data = num_new_data_run
        args.Q_update_time = q_update_time
        args.MAX_STEPS = 2 * args.T + args.num_update_time * args.num_new_data

        reward_save, cost_save = SLDAC_main(args, example_name)
        _save_mat_with_seed(
            build_algorithm_artifact_path(
                BASE_DIR,
                ALGORITHM_NAME,
                "SLDAC_reward_{0}.mat".format(output_suffix),
            ),
            {"array": reward_save},
            args,
            "SLDAC",
            output_suffix,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(
                BASE_DIR,
                ALGORITHM_NAME,
                "SLDAC_cost_{0}.mat".format(output_suffix),
            ),
            {"array": cost_save},
            args,
            "SLDAC",
            output_suffix,
        )


def main(args, example_name):
    experiment_seeds = resolve_experiment_seeds(args, DEFAULT_SEED)
    print("experiment seeds:", ", ".join(str(seed_value) for seed_value in experiment_seeds))
    for seed_value in experiment_seeds:
        print("run seed:", int(seed_value))
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed_value)
        _run_single_seed(seed_args, example_name)


example_name = "MIMO"
alpha_pow = 0.6
beta_pow = 0.7
eta_pow = 0.01
gamma_pow = 0.3
gamma_pow_reward = gamma_pow
gamma_pow_cost = gamma_pow
tau_reward = 1
tau_cost = 1

seed = DEFAULT_SEED
T = 500
num_new_data = 100
window = 10000
grad_T = T
episode = 60
update_time_per_episode = 10
num_update_time = episode * update_time_per_episode
Q_update_time = 1
MAX_STEPS = 2 * T + num_update_time * num_new_data
device = "cpu"
checkpoint_root = "checkpoints/SLDAC"
checkpoint_interval_episodes = 10
save_final_checkpoint = True


# 该入口以 .py 顶部配置为唯一配置源，CLI 仅保留帮助与兼容提示。
def build_python_config():
    return {
        "seed": int(seed),
        "seeds": _format_seed_list_text(DEFAULT_SEEDS),
        "T": int(T),
        "grad_T": int(grad_T),
        "window": int(window),
        "num_new_data": int(num_new_data),
        "episode": int(episode),
        "update_time_per_episode": int(update_time_per_episode),
        "num_update_time": int(num_update_time),
        "Q_update_time": int(Q_update_time),
        "MAX_STEPS": int(MAX_STEPS),
        "alpha_pow": float(alpha_pow),
        "beta_pow": float(beta_pow),
        "eta_pow": float(eta_pow),
        "gamma_pow_reward": float(gamma_pow_reward),
        "gamma_pow_cost": float(gamma_pow_cost),
        "tau_reward": float(tau_reward),
        "tau_cost": float(tau_cost),
        "device": str(device),
        "checkpoint_root": str(checkpoint_root),
        "checkpoint_interval_episodes": int(checkpoint_interval_episodes),
        "save_final_checkpoint": int(save_final_checkpoint),
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--T", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--grad_T", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_new_data", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--update_time_per_episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_update_time", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--Q_update_time", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--MAX_STEPS", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--alpha_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--beta_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eta_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma_pow_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma_pow_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_root", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint_interval_episodes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save_final_checkpoint", type=int, choices=[0, 1], default=argparse.SUPPRESS)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    cli_args = parser.parse_args()
    args, ignored_options = apply_python_config_priority(
        cli_args,
        build_python_config(),
        PROTECTED_CLI_FIELDS,
    )
    ignored_message = format_ignored_cli_overrides(ignored_options)
    if ignored_message:
        print(ignored_message)
    _migrate_legacy_checkpoints(args.checkpoint_root, example_name, default_seed=DEFAULT_SEED)
    main(args, example_name)
