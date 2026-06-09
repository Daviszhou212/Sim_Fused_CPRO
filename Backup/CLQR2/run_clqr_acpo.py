import argparse
import os
from collections import Counter

import numpy as np
from scipy.io import savemat

from ACPO import ACPO_main
from artifact_paths import build_algorithm_artifact_path
from seed_utils import (
    apply_python_config_priority,
    build_mat_metadata_from_args,
    format_ignored_cli_overrides,
    resolve_experiment_seeds,
)


EXAMPLE_NAME = "CLQR"
ALGORITHM_NAME = "ACPO"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认实验设置：论文口径优先，固定使用单约束 CLQR。
DEFAULT_SEED = 0
DEFAULT_SEEDS = (DEFAULT_SEED,)
DEFAULT_DEVICE = "cpu"
DEFAULT_STATE_DIM = 15
DEFAULT_ACTION_DIM = 4
DEFAULT_T = 500
DEFAULT_EPISODE = 101
DEFAULT_CONSTRAINT_LIMIT = 380.0
DEFAULT_GAE_LAMBDA_REWARD = 0.95
DEFAULT_GAE_LAMBDA_COST = 0.95
DEFAULT_DELTA = 1e-4
DEFAULT_BACKTRACK_COEFF = 0.75
DEFAULT_MAX_BACKTRACKS = 10
DEFAULT_CG_ITERS = 10
DEFAULT_CG_DAMPING = 1e-3
DEFAULT_RECOVERY_T = 0.75
DEFAULT_VF_LR = 2e-4
DEFAULT_VF_EPOCHS = 20
DEFAULT_VF_BATCH_SIZE = 256
DEFAULT_INIT_LOG_STD = -1.0
DEFAULT_FINAL_SUMMARY_TAIL = 20

ACPO_RUNS = [
    ("b500", "ACPO, T=500, batchsize=500", DEFAULT_T),
]

# 该入口以 .py 顶部配置为唯一配置源，CLI 仅保留帮助与兼容提示。
def build_python_config():
    return {
        "seed": int(DEFAULT_SEED),
        "seeds": None,
        "device": str(DEFAULT_DEVICE),
        "state_dim": int(DEFAULT_STATE_DIM),
        "action_dim": int(DEFAULT_ACTION_DIM),
        "T": int(DEFAULT_T),
        "episode": int(DEFAULT_EPISODE),
        "constraint_limit": float(DEFAULT_CONSTRAINT_LIMIT),
        "gae_lambda_reward": float(DEFAULT_GAE_LAMBDA_REWARD),
        "gae_lambda_cost": float(DEFAULT_GAE_LAMBDA_COST),
        "delta": float(DEFAULT_DELTA),
        "backtrack_coeff": float(DEFAULT_BACKTRACK_COEFF),
        "max_backtracks": int(DEFAULT_MAX_BACKTRACKS),
        "cg_iters": int(DEFAULT_CG_ITERS),
        "cg_damping": float(DEFAULT_CG_DAMPING),
        "recovery_t": float(DEFAULT_RECOVERY_T),
        "vf_lr": float(DEFAULT_VF_LR),
        "vf_epochs": int(DEFAULT_VF_EPOCHS),
        "vf_batch_size": int(DEFAULT_VF_BATCH_SIZE),
        "init_log_std": float(DEFAULT_INIT_LOG_STD),
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


def _build_mat_metadata(args, run_tag):
    return build_mat_metadata_from_args(args, ALGORITHM_NAME, run_tag, DEFAULT_SEED)


def _save_mat_with_seed(path, payload, args, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, run_tag))
    root, ext = os.path.splitext(str(path))
    seed_value = int(getattr(args, "seed", DEFAULT_SEED))
    savemat("{0}_seed{1}{2}".format(root, seed_value, ext), full_payload)


def _format_metric(value, precision=4):
    numeric = float(value)
    abs_numeric = abs(numeric)
    if (abs_numeric >= 1e4) or ((abs_numeric > 0.0) and (abs_numeric < 1e-3)):
        return "{0:.2e}".format(numeric)
    return "{0:.{1}f}".format(numeric, int(precision))


def _print_run_config(args, run_tag):
    print(
        "ACPO config | run_tag={0} | seed={1} | T={2} | episode={3} | constraint_limit={4} | "
        "delta={5} | gae_lambda_reward={6} | gae_lambda_cost={7} | recovery_t={8} | "
        "cg_iters={9} | cg_damping={10} | vf_lr={11} | vf_epochs={12} | vf_batch_size={13} | init_log_std={14}".format(
            str(run_tag),
            int(getattr(args, "seed", DEFAULT_SEED)),
            int(getattr(args, "T", DEFAULT_T)),
            int(getattr(args, "episode", DEFAULT_EPISODE)),
            _format_metric(getattr(args, "constraint_limit", DEFAULT_CONSTRAINT_LIMIT)),
            _format_metric(getattr(args, "delta", DEFAULT_DELTA), precision=6),
            _format_metric(getattr(args, "gae_lambda_reward", DEFAULT_GAE_LAMBDA_REWARD), precision=6),
            _format_metric(getattr(args, "gae_lambda_cost", DEFAULT_GAE_LAMBDA_COST), precision=6),
            _format_metric(getattr(args, "recovery_t", DEFAULT_RECOVERY_T), precision=6),
            int(getattr(args, "cg_iters", DEFAULT_CG_ITERS)),
            _format_metric(getattr(args, "cg_damping", DEFAULT_CG_DAMPING), precision=6),
            _format_metric(getattr(args, "vf_lr", DEFAULT_VF_LR), precision=6),
            int(getattr(args, "vf_epochs", DEFAULT_VF_EPOCHS)),
            int(getattr(args, "vf_batch_size", DEFAULT_VF_BATCH_SIZE)),
            _format_metric(getattr(args, "init_log_std", DEFAULT_INIT_LOG_STD), precision=6),
        )
    )


def _print_run_summary(objective_curve, cost_curve, diagnostics, run_tag):
    accepted = np.asarray(diagnostics["accepted"], dtype=np.int32).reshape(-1)
    modes = [str(item).strip() for item in np.asarray(diagnostics["mode"]).reshape(-1)]
    tail = min(int(DEFAULT_FINAL_SUMMARY_TAIL), len(objective_curve))
    mode_counts = Counter(modes)

    print(
        "ACPO summary | run_tag={0} | accepted_rate={1} | normal={2} | recovery={3} | skipped={4} | "
        "tail{5}_obj={6} | tail{5}_cost={7}".format(
            str(run_tag),
            _format_metric(float(np.mean(accepted)) if accepted.size > 0 else 0.0, precision=6),
            int(mode_counts.get("normal", 0)),
            int(mode_counts.get("recovery", 0)),
            int(mode_counts.get("skipped", 0)),
            int(tail),
            _format_metric(float(np.mean(objective_curve[-tail:])) if tail > 0 else 0.0),
            _format_metric(float(np.mean(cost_curve[-tail:])) if tail > 0 else 0.0),
        )
    )


def _run_single_seed(args):
    for run_tag, message, horizon in ACPO_RUNS:
        print(message)
        args.run_tag = run_tag
        args.T = int(horizon)
        _print_run_config(args, run_tag)
        reward_curve, cost_curve, diagnostics = ACPO_main(args, EXAMPLE_NAME)
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "ACPO_reward_{0}.mat".format(run_tag)),
            {"array": np.asarray(reward_curve, dtype=np.float64)},
            args,
            run_tag,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "ACPO_cost_{0}.mat".format(run_tag)),
            {"array": np.asarray(cost_curve, dtype=np.float64)},
            args,
            run_tag,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "ACPO_diag_{0}.mat".format(run_tag)),
            diagnostics,
            args,
            run_tag,
        )
        _print_run_summary(reward_curve, cost_curve, diagnostics, run_tag)


def main(args):
    experiment_seeds = resolve_experiment_seeds(args, DEFAULT_SEED)
    print("experiment seeds:", ", ".join(str(seed_value) for seed_value in experiment_seeds))
    for seed_value in experiment_seeds:
        print("run seed:", int(seed_value))
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed_value)
        _run_single_seed(seed_args)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--seeds", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--state_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--action_dim", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--T", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--constraint_limit", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gae_lambda_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gae_lambda_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--delta", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--backtrack_coeff", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--max_backtracks", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cg_iters", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--cg_damping", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--recovery_t", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--vf_lr", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--vf_epochs", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--vf_batch_size", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--init_log_std", type=float, default=argparse.SUPPRESS)
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
