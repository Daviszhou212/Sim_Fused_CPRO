import argparse
import os

import numpy as np
from scipy.io import savemat

from ACPO import ACPO_main
from artifact_paths import build_algorithm_artifact_path
from seed_utils import resolve_experiment_seeds


EXAMPLE_NAME = "CLQR"
ALGORITHM_NAME = "ACPO"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# 默认实验设置：论文口径优先，固定使用单约束 CLQR。
DEFAULT_SEED = 0
DEFAULT_SEEDS = (DEFAULT_SEED,)
DEFAULT_DEVICE = "cpu"
DEFAULT_STATE_DIM = 15
DEFAULT_ACTION_DIM = 4
DEFAULT_T = 250
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

ACPO_RUNS = [
    ("b250", "ACPO, batchsize=250", DEFAULT_T),
]


def _build_mat_metadata(args, run_tag):
    return {
        "seed": np.asarray([[int(getattr(args, "seed", DEFAULT_SEED))]], dtype=np.int32),
        "algorithm": np.asarray([ALGORITHM_NAME], dtype="U32"),
        "run_tag": np.asarray([str(run_tag)], dtype="U32"),
    }


def _save_mat_with_seed(path, payload, args, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, run_tag))
    root, ext = os.path.splitext(str(path))
    seed_value = int(getattr(args, "seed", DEFAULT_SEED))
    savemat("{0}_seed{1}{2}".format(root, seed_value, ext), full_payload)


def _run_single_seed(args):
    for run_tag, message, horizon in ACPO_RUNS:
        print(message)
        args.run_tag = run_tag
        args.T = int(horizon)
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
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--seeds", type=str, default=None)
    parser.add_argument("--device", type=str, default=DEFAULT_DEVICE)
    parser.add_argument("--state_dim", type=int, default=DEFAULT_STATE_DIM)
    parser.add_argument("--action_dim", type=int, default=DEFAULT_ACTION_DIM)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--episode", type=int, default=DEFAULT_EPISODE)
    parser.add_argument("--constraint_limit", type=float, default=DEFAULT_CONSTRAINT_LIMIT)
    parser.add_argument("--gae_lambda_reward", type=float, default=DEFAULT_GAE_LAMBDA_REWARD)
    parser.add_argument("--gae_lambda_cost", type=float, default=DEFAULT_GAE_LAMBDA_COST)
    parser.add_argument("--delta", type=float, default=DEFAULT_DELTA)
    parser.add_argument("--backtrack_coeff", type=float, default=DEFAULT_BACKTRACK_COEFF)
    parser.add_argument("--max_backtracks", type=int, default=DEFAULT_MAX_BACKTRACKS)
    parser.add_argument("--cg_iters", type=int, default=DEFAULT_CG_ITERS)
    parser.add_argument("--cg_damping", type=float, default=DEFAULT_CG_DAMPING)
    parser.add_argument("--recovery_t", type=float, default=DEFAULT_RECOVERY_T)
    parser.add_argument("--vf_lr", type=float, default=DEFAULT_VF_LR)
    parser.add_argument("--vf_epochs", type=int, default=DEFAULT_VF_EPOCHS)
    parser.add_argument("--vf_batch_size", type=int, default=DEFAULT_VF_BATCH_SIZE)
    parser.add_argument("--init_log_std", type=float, default=DEFAULT_INIT_LOG_STD)
    return parser


if __name__ == "__main__":
    parser = build_parser()
    main(parser.parse_args())
