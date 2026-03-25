import argparse
import os

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from artifact_paths import build_algorithm_artifact_path
from seed_utils import apply_python_config_priority, format_ignored_cli_overrides, resolve_experiment_seeds
from Fused_CPRO import PRCRL_main, _resolve_sldac_checkpoint_path
from run_clqr_sldac import _migrate_legacy_checkpoints


PRCRL_RUNS = [
    ("default", "PRCRL, no offline data", 250, 250, 100, 5),
]

DEFAULT_SEED = 0
DEFAULT_WINDOW = 10000
DEFAULT_EPISODE = 101
DEFAULT_UPDATE_TIME_PER_EPISODE = 10
DEFAULT_NUM_UPDATE_TIME = DEFAULT_EPISODE * DEFAULT_UPDATE_TIME_PER_EPISODE
DEFAULT_ALPHA_POW = 0.6
DEFAULT_BETA_POW = 0.8
DEFAULT_BETA_ACTOR_POW = DEFAULT_BETA_POW
DEFAULT_BETA_RHO_POW = 0.9
DEFAULT_ETA_POW = 0.01
DEFAULT_GAMMA_POW_REWARD = 0.27
DEFAULT_GAMMA_POW_COST = 0.27
DEFAULT_TAU_REWARD = 6.0
DEFAULT_TAU_COST = 10.0
DEFAULT_DEVICE = "cpu"
DEFAULT_OLD_POLICY_SEED = 1
DEFAULT_OLD_POLICY_CHECKPOINT_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "checkpoints", "SLDAC")

EXAMPLE_NAME = "CLQR"
ALGORITHM_NAME = "PRCRL"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OLD_POLICY_BQ_LIST = [(100, 10)]
OLD_POLICY_PRETRAIN_EPISODE = 50
OLD_POLICY_CHECKPOINT_ROOT = DEFAULT_OLD_POLICY_CHECKPOINT_ROOT


def build_python_config():
    return {
        "seed": int(DEFAULT_SEED),
        "seeds": None,
        "window": int(DEFAULT_WINDOW),
        "episode": int(DEFAULT_EPISODE),
        "update_time_per_episode": int(DEFAULT_UPDATE_TIME_PER_EPISODE),
        "num_update_time": int(DEFAULT_NUM_UPDATE_TIME),
        "alpha_pow": float(DEFAULT_ALPHA_POW),
        "beta_pow": float(DEFAULT_BETA_POW),
        "beta_actor_pow": float(DEFAULT_BETA_ACTOR_POW),
        "beta_rho_pow": float(DEFAULT_BETA_RHO_POW),
        "eta_pow": float(DEFAULT_ETA_POW),
        "gamma_pow_reward": float(DEFAULT_GAMMA_POW_REWARD),
        "gamma_pow_cost": float(DEFAULT_GAMMA_POW_COST),
        "tau_reward": float(DEFAULT_TAU_REWARD),
        "tau_cost": float(DEFAULT_TAU_COST),
        "device": str(DEFAULT_DEVICE),
        "old_policies": None,
        "old_policy_seed": int(DEFAULT_OLD_POLICY_SEED),
        "old_policy_pretrain_episode": int(OLD_POLICY_PRETRAIN_EPISODE),
        "old_policy_checkpoint_root": str(OLD_POLICY_CHECKPOINT_ROOT),
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


def _parse_positive_int(value, field_name, source_text):
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise ValueError(
            "invalid {0} in old policy spec {1!r}: expected a positive integer.".format(field_name, source_text)
        )
    if parsed <= 0:
        raise ValueError(
            "invalid {0} in old policy spec {1!r}: expected a positive integer.".format(field_name, source_text)
        )
    return parsed


def _format_old_policy_run_tag(batch_size, q_update_time):
    return "b{0}_q{1}".format(int(batch_size), int(q_update_time))


def _dedupe_run_tags(run_tags):
    normalized = []
    seen = set()
    for run_tag in run_tags:
        if run_tag not in seen:
            seen.add(run_tag)
            normalized.append(run_tag)
    return normalized


def _parse_old_policy_cli(old_policies_text):
    text = "" if old_policies_text is None else str(old_policies_text).strip()
    if not text:
        return []

    run_tags = []
    for raw_spec in text.split(","):
        spec = raw_spec.strip()
        if not spec:
            continue
        parts = spec.split(":")
        if len(parts) != 2:
            raise ValueError("invalid --old-policies spec {0!r}. expected format like b100:q1,b500:q10".format(spec))
        batch_part, q_part = parts
        if (len(batch_part) <= 1) or (batch_part[0].lower() != "b"):
            raise ValueError("invalid --old-policies spec {0!r}. expected the batch part to look like b100".format(spec))
        if (len(q_part) <= 1) or (q_part[0].lower() != "q"):
            raise ValueError("invalid --old-policies spec {0!r}. expected the q part to look like q1".format(spec))
        batch_size = _parse_positive_int(batch_part[1:], "b", spec)
        q_update_time = _parse_positive_int(q_part[1:], "q", spec)
        run_tags.append(_format_old_policy_run_tag(batch_size, q_update_time))
    return _dedupe_run_tags(run_tags)


def _normalize_old_policy_bq_list(bq_list):
    run_tags = []
    for item in bq_list:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("OLD_POLICY_BQ_LIST item must be a (b, q) pair. got {0!r}".format(item))
        batch_size = _parse_positive_int(item[0], "b", item)
        q_update_time = _parse_positive_int(item[1], "q", item)
        run_tags.append(_format_old_policy_run_tag(batch_size, q_update_time))
    return _dedupe_run_tags(run_tags)


def _resolve_old_policy_args(args):
    args.old_policy_seed = int(getattr(args, "old_policy_seed", DEFAULT_OLD_POLICY_SEED))
    if getattr(args, "old_policies", None) is None:
        run_tags = _normalize_old_policy_bq_list(OLD_POLICY_BQ_LIST)
    else:
        run_tags = _parse_old_policy_cli(args.old_policies)

    if getattr(args, "old_policy_pretrain_episode", None) is None:
        pretrain_episode = int(OLD_POLICY_PRETRAIN_EPISODE)
    else:
        pretrain_episode = int(args.old_policy_pretrain_episode)

    checkpoint_root = getattr(args, "old_policy_checkpoint_root", None) or OLD_POLICY_CHECKPOINT_ROOT
    args.old_policy_run_tags = ",".join(run_tags)
    args.old_policy_pretrain_episode = pretrain_episode
    args.old_policy_checkpoint_root = checkpoint_root
    args.pretrain_episode = pretrain_episode
    args.checkpoint_root = checkpoint_root
    if run_tags and (pretrain_episode <= 0):
        raise ValueError(
            "old policy pretrain_episode must be a positive integer when old policies are configured. got {0}".format(
                pretrain_episode
            )
        )
    return args


def _validate_old_policy_checkpoints(args):
    run_tags = [tag.strip() for tag in str(getattr(args, "old_policy_run_tags", "")).split(",") if tag.strip()]
    if not run_tags:
        return args

    print("selected old policy run_tags:", ", ".join(run_tags))
    print("selected old policy seed:", int(args.old_policy_seed))
    print("selected old policy pretrain_episode:", int(args.old_policy_pretrain_episode))
    for run_tag in run_tags:
        checkpoint_path = _resolve_sldac_checkpoint_path(
            args,
            EXAMPLE_NAME,
            run_tag,
            int(args.old_policy_pretrain_episode),
            int(args.old_policy_seed),
        )
        print("verified old policy checkpoint:", run_tag, "->", checkpoint_path)
    return args


def _finalize_actor_rho_args(args):
    if getattr(args, "beta_actor_pow", None) is None:
        args.beta_actor_pow = float(getattr(args, "beta_pow", DEFAULT_BETA_POW))
    else:
        args.beta_actor_pow = float(args.beta_actor_pow)

    if getattr(args, "beta_rho_pow", None) is None:
        args.beta_rho_pow = float(DEFAULT_BETA_RHO_POW)
    else:
        args.beta_rho_pow = float(args.beta_rho_pow)

    if float(args.beta_rho_pow) <= float(args.beta_actor_pow):
        raise ValueError(
            "beta_rho_pow must be greater than beta_actor_pow. got beta_actor_pow={0}, beta_rho_pow={1}".format(
                args.beta_actor_pow,
                args.beta_rho_pow,
            )
        )
    return args


def _plot_reuse_probability(output_suffix, rho_history, rho_labels, seed):
    if rho_history.size == 0:
        return
    out_path = build_algorithm_artifact_path(
        BASE_DIR,
        ALGORITHM_NAME,
        "PRCRL_reuse_prob_{0}_seed{1}.png".format(output_suffix, int(seed)),
    )
    x = np.arange(1, rho_history.shape[0] + 1)
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.6))
    for idx, label in enumerate(rho_labels):
        ax.plot(x, rho_history[:, idx], linewidth=2.0, label=label)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Reuse probability")
    ax.set_title("CLQR PRCRL reuse probabilities: {0}".format(output_suffix))
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _moving_average(values, window=5):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return arr
    out = np.zeros_like(arr)
    for idx in range(arr.size):
        left = max(0, idx - int(window) + 1)
        out[idx] = np.mean(arr[left : idx + 1])
    return out


def _refresh_max_steps(args):
    args.MAX_STEPS = 2 * int(args.T) + int(args.num_update_time) * int(args.num_new_data)
    return args


def _apply_run_config(args, run_tag, message, t_horizon, grad_t, num_new_data, q_update_time):
    print(message)
    args.run_tag = run_tag
    args.T = int(t_horizon)
    args.grad_T = int(grad_t)
    args.num_new_data = int(num_new_data)
    args.Q_update_time = int(q_update_time)
    return _refresh_max_steps(args)


def _collect_drift_series(drift_history):
    series = []
    for key, label in (("actor_rms", "actor"), ("critic_rms", "critic"), ("rho_rms", "rho")):
        values = np.asarray(drift_history.get(key, []), dtype=np.float64).reshape(-1)
        if values.size > 0:
            series.append((values, label))
    return series


def _plot_drift_speed(output_suffix, drift_history, seed):
    update_index = np.asarray(drift_history.get("update_index", []), dtype=np.float64).reshape(-1)
    if update_index.size <= 0:
        return
    series = _collect_drift_series(drift_history)
    if not series:
        return
    out_path = build_algorithm_artifact_path(
        BASE_DIR,
        ALGORITHM_NAME,
        "PRCRL_drift_speed_{0}_seed{1}.png".format(output_suffix, int(seed)),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    for values, label in series:
        axes[0].plot(update_index, values, linewidth=2.0, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS drift")
    axes[0].set_title("CLQR PRCRL drift speeds: {0}".format(output_suffix))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=max(1, len(series)))
    for values, label in series:
        axes[1].plot(update_index, _moving_average(values, window=5), linewidth=2.0, label=label)
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Policy update")
    axes[1].set_ylabel("RMS drift (MA5)")
    axes[1].grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


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
    parser.add_argument("--beta_actor_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--beta_rho_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eta_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma_pow_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma_pow_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--old-policies", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--old-policy-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--old-policy-pretrain-episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--old-policy-checkpoint-root", type=str, default=argparse.SUPPRESS)
    return parser


def _run_single_seed(args):
    for run_tag, message, t_horizon, grad_t, num_new_data, q_update_time in PRCRL_RUNS:
        run_args = argparse.Namespace(**vars(args))
        run_args = _apply_run_config(run_args, run_tag, message, t_horizon, grad_t, num_new_data, q_update_time)
        reward_save, cost_save, rho_history, _, rho_labels, drift_history = PRCRL_main(
            run_args,
            EXAMPLE_NAME,
            return_aux=True,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "PRCRL_reward_{0}.mat".format(run_tag)),
            {"array": reward_save},
            run_args,
            ALGORITHM_NAME,
            run_tag,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "PRCRL_cost_{0}.mat".format(run_tag)),
            {"array": cost_save},
            run_args,
            ALGORITHM_NAME,
            run_tag,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "PRCRL_rho_{0}.mat".format(run_tag)),
            {"array": rho_history, "labels": np.asarray(rho_labels, dtype="U32")},
            run_args,
            ALGORITHM_NAME,
            run_tag,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, "PRCRL_drift_{0}.mat".format(run_tag)),
            drift_history,
            run_args,
            ALGORITHM_NAME,
            run_tag,
        )
        _plot_reuse_probability(run_tag, rho_history, rho_labels, run_args.seed)
        _plot_drift_speed(run_tag, drift_history, run_args.seed)


def main():
    parser = build_parser()
    cli_args = parser.parse_args()
    args, ignored_options = apply_python_config_priority(cli_args, build_python_config(), PROTECTED_CLI_FIELDS)
    ignored_message = format_ignored_cli_overrides(ignored_options)
    if ignored_message:
        print(ignored_message)
    args = _finalize_actor_rho_args(args)
    args = _resolve_old_policy_args(args)
    _migrate_legacy_checkpoints(args.checkpoint_root, EXAMPLE_NAME, default_seed=int(args.old_policy_seed))
    args = _validate_old_policy_checkpoints(args)
    experiment_seeds = resolve_experiment_seeds(args, DEFAULT_SEED)
    print("experiment seeds:", ", ".join(str(seed_value) for seed_value in experiment_seeds))
    for seed_value in experiment_seeds:
        print("run seed:", int(seed_value))
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed_value)
        _run_single_seed(seed_args)


if __name__ == "__main__":
    main()
