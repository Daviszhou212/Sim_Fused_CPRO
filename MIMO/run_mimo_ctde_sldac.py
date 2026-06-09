import argparse
import os

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


# 固定场景：CTDE 多小区 MIMO 的 SLDAC 原型入口。
EXAMPLE_NAME = "MIMO_CTDE"
ALGORITHM_NAME = "SLDAC_CTDE"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 多小区拓扑配置；先用 3 小区、每小区 2 用户，便于验证跨小区干扰链路。
DEFAULT_NT = 8
DEFAULT_NUM_CELLS = 3
DEFAULT_USERS_PER_CELL = 2
DEFAULT_CONSTRAINT_LIMIT = 1.2

# 初版 CTDE smoke 规模；正式实验可直接在本文件顶部改大。
SLDAC_CTDE_RUNS = [
    ("ctde_c3_k2_b50_q1", "CTDE SLDAC, 3 cells, 2 users/cell, batchsize=50, q=1", 50, 50, 20, 1),
]

DEFAULT_SEED = 0
DEFAULT_SEEDS = (DEFAULT_SEED,)
DEFAULT_WINDOW = 10000
DEFAULT_EPISODE = 20
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
DEFAULT_ACTOR_DISTRIBUTION = "squashed"
DEFAULT_CHECKPOINT_ROOT = "checkpoints/SLDAC_CTDE"
DEFAULT_CHECKPOINT_INTERVAL_EPISODES = 10
DEFAULT_SAVE_FINAL_CHECKPOINT = True


def _format_seed_list_text(seed_values):
    return ",".join(str(int(seed_value)) for seed_value in seed_values)


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


def _build_mat_metadata(args, algorithm, run_tag):
    return build_mat_metadata_from_args(args, algorithm, run_tag, DEFAULT_SEED)


def _save_mat_with_seed(filename, payload, args, algorithm, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, algorithm, run_tag))
    root, ext = os.path.splitext(str(filename))
    seed_value = int(getattr(args, "seed", DEFAULT_SEED))
    savemat("{0}_seed{1}{2}".format(root, seed_value, ext), full_payload)


def _run_single_seed(args):
    for output_suffix, message, t_horizon, grad_t, num_new_data_run, q_update_time in SLDAC_CTDE_RUNS:
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
                "SLDAC_CTDE_reward_{0}.mat".format(output_suffix),
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
                "SLDAC_CTDE_cost_{0}.mat".format(output_suffix),
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
        "Nt": int(DEFAULT_NT),
        "num_cells": int(DEFAULT_NUM_CELLS),
        "users_per_cell": int(DEFAULT_USERS_PER_CELL),
        "constraint_limit": float(DEFAULT_CONSTRAINT_LIMIT),
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
        "actor_distribution": str(DEFAULT_ACTOR_DISTRIBUTION),
        "checkpoint_root": str(DEFAULT_CHECKPOINT_ROOT),
        "checkpoint_interval_episodes": int(DEFAULT_CHECKPOINT_INTERVAL_EPISODES),
        "save_final_checkpoint": int(DEFAULT_SAVE_FINAL_CHECKPOINT),
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--Nt", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-cells", dest="num_cells", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--users-per-cell", dest="users_per_cell", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--constraint-limit", dest="constraint_limit", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--window", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--update-time-per-episode", dest="update_time_per_episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--num-update-time", dest="num_update_time", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--alpha-pow", dest="alpha_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--beta-pow", dest="beta_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eta-pow", dest="eta_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma-pow-reward", dest="gamma_pow_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma-pow-cost", dest="gamma_pow_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau-reward", dest="tau_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau-cost", dest="tau_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--actor-distribution", dest="actor_distribution", type=str, choices=["squashed", "legacy"], default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-root", dest="checkpoint_root", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--checkpoint-interval-episodes", dest="checkpoint_interval_episodes", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--save-final-checkpoint", dest="save_final_checkpoint", type=int, choices=[0, 1], default=argparse.SUPPRESS)
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
    main(args)
