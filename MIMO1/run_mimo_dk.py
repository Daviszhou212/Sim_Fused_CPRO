import argparse
import os

import numpy as np
from scipy.io import savemat

from artifact_paths import build_algorithm_artifact_path
from seed_utils import apply_python_config_priority, format_ignored_cli_overrides, resolve_experiment_seeds
from Fused_CPRO import DK_main


EXAMPLE_NAME = "MIMO"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "cpu"
ALGORITHM_NAME = "DK"

# 固定实验组：默认与 MIMO Fused-CPRO/HRL 对齐，保证 compare 图直接可对比。
DK_RUNS = [
    ("b100_q1", "DK, fixed policy, batchsize=100, q=1", 500, 500, 100, 1),
    # ("b100_q5", "DK, fixed policy, batchsize=100, q=5", 500, 500, 100, 5),
    # ("b100_q10", "DK, fixed policy, batchsize=100, q=10", 500, 500, 100, 10),
    # ("b500_q10", "DK, fixed policy, T=500, batchsize=500, q=10", 50, 100, 100, 10),
]

# 顶部可调配置：DK 基线直接复用现有 MIMO 主实验的记点节奏。
DEFAULT_SEED = 0
DEFAULT_SEEDS = (DEFAULT_SEED,)
DEFAULT_WINDOW = 10000
DEFAULT_EPISODE = 60
DEFAULT_UPDATE_TIME_PER_EPISODE = 10
DEFAULT_NUM_UPDATE_TIME = DEFAULT_EPISODE * DEFAULT_UPDATE_TIME_PER_EPISODE


def build_python_config():
    return {
        "seed": int(DEFAULT_SEED),
        "seeds": ",".join(str(int(seed_value)) for seed_value in DEFAULT_SEEDS),
        "window": int(DEFAULT_WINDOW),
        "episode": int(DEFAULT_EPISODE),
        "update_time_per_episode": int(DEFAULT_UPDATE_TIME_PER_EPISODE),
        "num_update_time": int(DEFAULT_NUM_UPDATE_TIME),
        "device": str(DEVICE),
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


def _build_mat_metadata(args, algorithm, run_tag):
    return {
        "seed": np.asarray([[int(getattr(args, "seed", DEFAULT_SEED))]], dtype=np.int32),
        "algorithm": np.asarray([str(algorithm)], dtype="U32"),
        "run_tag": np.asarray([str(run_tag)], dtype="U32"),
    }


def _save_mat_with_seed(path, payload, args, algorithm, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, algorithm, run_tag))
    root, ext = os.path.splitext(str(path))
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
    args = _refresh_max_steps(args)
    return args


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--update_time_per_episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num_update_time", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    return parser


def _run_single_seed(args):
    for output_suffix, message, t_horizon, grad_t, num_new_data_run, q_update_time in DK_RUNS:
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
        reward_save, cost_save = DK_main(run_args, EXAMPLE_NAME)
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "DK_reward_{0}.mat".format(output_suffix)),
            {"array": reward_save},
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "DK_cost_{0}.mat".format(output_suffix)),
            {"array": cost_save},
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )


def main():
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
    experiment_seeds = resolve_experiment_seeds(args, DEFAULT_SEED)
    print("experiment seeds:", ", ".join(str(seed_value) for seed_value in experiment_seeds))
    for seed_value in experiment_seeds:
        print("run seed:", int(seed_value))
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed_value)
        _run_single_seed(seed_args)


if __name__ == "__main__":
    main()
