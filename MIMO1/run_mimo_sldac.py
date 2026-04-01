import argparse
import os
import shutil

import numpy as np
from scipy.io import savemat

from artifact_paths import build_algorithm_artifact_path
from seed_utils import (
    apply_python_config_priority,
    build_mat_metadata_from_args,
    format_ignored_cli_overrides,
    resolve_experiment_seeds,
)
from SLDAC import SLDAC_main


# Fixed SLDAC runs kept for the current MIMO reproduction.
# Tuple fields: output suffix, log label, T, grad_T, num_new_data, Q_update_time.
SLDAC_RUNS = [
    ("b100_q1", "SLDAC, batchsize=100, q=1", 500, 500, 100, 1),
    # ("b100_q5", "SLDAC, batchsize=100, q=5", 500, 500, 100, 5),
    # ("b100_q10", "SLDAC, batchsize=100, q=10", 500, 500, 100, 10),
    # ("b500_q10", "SLDAC, T=500, batchsize=500, q=10", 50, 100, 100, 10),
]

EXAMPLE_NAME = "MIMO"
ALGORITHM_NAME = "SLDAC"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Top-level defaults shared by all active SLDAC runs in this entry script.
DEFAULT_SEED = 0
DEFAULT_SEEDS = (0,)
DEFAULT_WINDOW = 10000
DEFAULT_EPISODE = 100
DEFAULT_UPDATE_TIME_PER_EPISODE = 10
DEFAULT_NUM_UPDATE_TIME = DEFAULT_EPISODE * DEFAULT_UPDATE_TIME_PER_EPISODE
DEFAULT_ALPHA_POW = 0.6
DEFAULT_BETA_POW = 0.7
DEFAULT_ETA_POW = 0.01
DEFAULT_GAMMA_POW_REWARD = 0.3
DEFAULT_GAMMA_POW_COST = 0.3
DEFAULT_TAU_REWARD = 1.0
DEFAULT_TAU_COST = 1.0
DEFAULT_DEVICE = "cpu"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/SLDAC"
DEFAULT_CHECKPOINT_INTERVAL_EPISODES = 10
DEFAULT_SAVE_FINAL_CHECKPOINT = True

LEGACY_CHECKPOINT_PREFIX = "episode_"
LEGACY_CHECKPOINT_SUFFIX = ".pt"


def _format_seed_dir(seed):
    return "seed_{0}".format(int(seed))


def _format_seed_list_text(seed_values):
    return ",".join(str(int(seed_value)) for seed_value in seed_values)


def _resolve_checkpoint_root(checkpoint_root):
    root = checkpoint_root or DEFAULT_CHECKPOINT_ROOT
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
    return build_mat_metadata_from_args(args, algorithm, run_tag, DEFAULT_SEED)


def _save_mat_with_seed(filename, payload, args, algorithm, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, algorithm, run_tag))
    root, ext = os.path.splitext(str(filename))
    seed_value = int(getattr(args, "seed", DEFAULT_SEED))
    savemat("{0}_seed{1}{2}".format(root, seed_value, ext), full_payload)


def _refresh_max_steps(args):
    args.MAX_STEPS = 2 * int(args.T) + int(args.num_update_time) * int(args.num_new_data)
    return args


def _apply_run_config(args, output_suffix, message, t_horizon, grad_t, num_new_data_run, q_update_time):
    print(message)
    args.run_tag = output_suffix
    args.T = int(t_horizon)
    args.grad_T = int(grad_t)
    args.num_new_data = int(num_new_data_run)
    args.Q_update_time = int(q_update_time)
    return _refresh_max_steps(args)


def _run_single_seed(args):
    for output_suffix, message, t_horizon, grad_t, num_new_data_run, q_update_time in SLDAC_RUNS:
        run_args = argparse.Namespace(**vars(args))
        run_args = _apply_run_config(
            run_args,
            output_suffix,
            message,
            t_horizon,
            grad_t,
            num_new_data_run,
            q_update_time,
        )

        reward_save, cost_save = SLDAC_main(run_args, EXAMPLE_NAME)
        _save_mat_with_seed(
            build_algorithm_artifact_path(
                BASE_DIR,
                ALGORITHM_NAME,
                "SLDAC_reward_{0}.mat".format(output_suffix),
            ),
            {"array": reward_save},
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(
                BASE_DIR,
                ALGORITHM_NAME,
                "SLDAC_cost_{0}.mat".format(output_suffix),
            ),
            {"array": cost_save},
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )


def main(args):
    experiment_seeds = resolve_experiment_seeds(args, DEFAULT_SEED)
    print("experiment seeds:", ", ".join(str(seed_value) for seed_value in experiment_seeds))
    for seed_value in experiment_seeds:
        print("run seed:", int(seed_value))
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed_value)
        _run_single_seed(seed_args)


def build_python_config():
    return {
        "seed": int(DEFAULT_SEED),
        "seeds": _format_seed_list_text(DEFAULT_SEEDS),
        "window": int(DEFAULT_WINDOW),
        "episode": int(DEFAULT_EPISODE),
        "update_time_per_episode": int(DEFAULT_UPDATE_TIME_PER_EPISODE),
        "num_update_time": int(DEFAULT_NUM_UPDATE_TIME),
        "alpha_pow": float(DEFAULT_ALPHA_POW),
        "beta_pow": float(DEFAULT_BETA_POW),
        "eta_pow": float(DEFAULT_ETA_POW),
        "gamma_pow_reward": float(DEFAULT_GAMMA_POW_REWARD),
        "gamma_pow_cost": float(DEFAULT_GAMMA_POW_COST),
        "tau_reward": float(DEFAULT_TAU_REWARD),
        "tau_cost": float(DEFAULT_TAU_COST),
        "device": str(DEFAULT_DEVICE),
        "checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT),
        "checkpoint_interval_episodes": int(DEFAULT_CHECKPOINT_INTERVAL_EPISODES),
        "save_final_checkpoint": int(DEFAULT_SAVE_FINAL_CHECKPOINT),
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--update_time_per_episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_update_time", type=int, default=argparse.SUPPRESS)
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
    _migrate_legacy_checkpoints(args.checkpoint_root, EXAMPLE_NAME, default_seed=DEFAULT_SEED)
    main(args)
