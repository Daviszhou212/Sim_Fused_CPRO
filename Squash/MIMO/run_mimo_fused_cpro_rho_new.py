import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat

from artifact_paths import build_algorithm_artifact_path
from Fused_CPRO import Fused_CPRO_RhoNew_main, RHO_SCHEDULER_EPISODE_PEAK_EXP_DECAY, _resolve_sldac_checkpoint_path
from run_mimo_sldac import _migrate_legacy_checkpoints
from seed_utils import (
    apply_python_config_priority,
    build_mat_metadata_from_args,
    format_ignored_cli_overrides,
    resolve_experiment_seeds,
)


EXAMPLE_NAME = "MIMO"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEVICE = "auto"
ALGORITHM_NAME = "Fused_CPRO_RhoNew"
ALGORITHM_LABEL = "Fused-CPRO-RhoNew"
RHO_SCHEDULER = RHO_SCHEDULER_EPISODE_PEAK_EXP_DECAY

# 固定实验组：默认与 MIMO Fused-CPRO 保持同一组 horizon / q 配置。
FUSED_CPRO_RHONEW_RUNS = [
    ("b100_q1", "{0}, batchsize=100, q=1".format(ALGORITHM_LABEL), 500, 500, 100, 1),
    # ("b100_q5", "{0}, batchsize=100, q=5".format(ALGORITHM_LABEL), 500, 500, 100, 5),
    # ("b100_q10", "{0}, batchsize=100, q=10".format(ALGORITHM_LABEL), 500, 500, 100, 10),
    # ("b500_q10", "{0}, T=500, batchsize=500, q=10".format(ALGORITHM_LABEL), 50, 100, 100, 10),
]

# 顶部可调超参数：基础参数与 MIMO Fused-CPRO 对齐。
DEFAULT_SEED = 0
DEFAULT_WINDOW = 10000
DEFAULT_EPISODE = 60
DEFAULT_UPDATE_TIME_PER_EPISODE = 10
DEFAULT_NUM_UPDATE_TIME = DEFAULT_EPISODE * DEFAULT_UPDATE_TIME_PER_EPISODE
DEFAULT_ALPHA_POW = 0.6
DEFAULT_BETA_ACTOR_POW = 0.7
DEFAULT_BETA_RHO_POW = 0.8
# xi0 表示 offline 分支权重；0.5 表示 online/offline 两个分支各占一半。
DEFAULT_XI0 = 1
DEFAULT_ACTOR_XI0 = None
DEFAULT_CRITIC_XI0 = None
# xi_pow 表示 xi 的幂次衰减系数，值越大代表离线权重下降越快。
DEFAULT_XI_POW = 0.95
DEFAULT_ACTOR_XI_POW = None
DEFAULT_CRITIC_XI_POW = None
# critic超参
DEFAULT_ETA_POW = 0.01
DEFAULT_GAMMA_POW_REWARD = 0.3
DEFAULT_GAMMA_POW_COST = 0.3
# 二次强凸项系数
DEFAULT_TAU_REWARD = 1.0
DEFAULT_TAU_COST = 1.0
DEFAULT_CSSCA_SOLVER = "cvx"
DEFAULT_RHO_MIN_NEW_ACTOR = 1e-4
DEFAULT_RHO_MIN_OLD_POLICY = 1e-4
DEFAULT_FREEZE_RHO_EPISODE_COUNT = 0
DEFAULT_ACTOR_DISTRIBUTION = "squashed"

# CosRho 额外调度参数：控制 rho 的 cosine-restart-decay 更新。

DEFAULT_RHO_BETA_PEAK_EPISODE = 5
DEFAULT_RHO_BETA_PEAK_VALUE = 1
DEFAULT_RHO_BETA_END_VALUE = 0.005

# old policy 配置：显式选择要复用的 SLDAC checkpoint 组。
DEFAULT_OLD_POLICY_SEED = 1
OLD_POLICY_BQ_LIST = [(100, 1)]
OLD_POLICY_PRETRAIN_EPISODE = 40
OLD_POLICY_CHECKPOINT_ROOT = os.path.join(BASE_DIR, "checkpoints", "SLDAC")
NEW_POLICY_INIT_BQ = (100, 10)
NEW_POLICY_INIT_SEED = 0
NEW_POLICY_INIT_PRETRAIN_EPISODE = 10
NEW_POLICY_INIT_CHECKPOINT_ROOT = OLD_POLICY_CHECKPOINT_ROOT
LOAD_NEW_ACTOR = True


def build_python_config():
    return {
        "seed": int(DEFAULT_SEED),
        "seeds": None,
        "window": int(DEFAULT_WINDOW),
        "episode": int(DEFAULT_EPISODE),
        "update_time_per_episode": int(DEFAULT_UPDATE_TIME_PER_EPISODE),
        "num_update_time": int(DEFAULT_NUM_UPDATE_TIME),
        "alpha_pow": float(DEFAULT_ALPHA_POW),
        "beta_actor_pow": float(DEFAULT_BETA_ACTOR_POW),
        "beta_rho_pow": float(DEFAULT_BETA_RHO_POW),
        "xi0": float(DEFAULT_XI0),
        "actor_xi0": DEFAULT_ACTOR_XI0,
        "critic_xi0": DEFAULT_CRITIC_XI0,
        "xi_pow": float(DEFAULT_XI_POW),
        "actor_xi_pow": DEFAULT_ACTOR_XI_POW,
        "critic_xi_pow": DEFAULT_CRITIC_XI_POW,
        "eta_pow": float(DEFAULT_ETA_POW),
        "gamma_pow_reward": float(DEFAULT_GAMMA_POW_REWARD),
        "gamma_pow_cost": float(DEFAULT_GAMMA_POW_COST),
        "tau_reward": float(DEFAULT_TAU_REWARD),
        "tau_cost": float(DEFAULT_TAU_COST),
        "cssca_solver": str(DEFAULT_CSSCA_SOLVER),
        "rho_min_new_actor": float(DEFAULT_RHO_MIN_NEW_ACTOR),
        "rho_min_old_policy": float(DEFAULT_RHO_MIN_OLD_POLICY),
        "freeze_rho_episode_count": int(DEFAULT_FREEZE_RHO_EPISODE_COUNT),
        "device": str(DEVICE),
        "actor_distribution": str(DEFAULT_ACTOR_DISTRIBUTION),
        "rho_scheduler": str(RHO_SCHEDULER),
        "rho_beta_peak_episode": int(DEFAULT_RHO_BETA_PEAK_EPISODE),
        "rho_beta_peak_value": float(DEFAULT_RHO_BETA_PEAK_VALUE),
        "rho_beta_end_value": float(DEFAULT_RHO_BETA_END_VALUE),
        "old_policies": None,
        "old_policy_seed": int(DEFAULT_OLD_POLICY_SEED),
        "old_policy_pretrain_episode": int(OLD_POLICY_PRETRAIN_EPISODE),
        "old_policy_checkpoint_root": str(OLD_POLICY_CHECKPOINT_ROOT),
        "load_new_actor": bool(LOAD_NEW_ACTOR),
        "new_policy_init": NEW_POLICY_INIT_BQ,
        "new_policy_seed": int(NEW_POLICY_INIT_SEED),
        "new_policy_pretrain_episode": int(NEW_POLICY_INIT_PRETRAIN_EPISODE),
        "new_policy_checkpoint_root": str(NEW_POLICY_INIT_CHECKPOINT_ROOT),
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


def _build_mat_metadata(args, algorithm, run_tag):
    return build_mat_metadata_from_args(args, algorithm, run_tag, DEFAULT_SEED)


def _save_mat_with_seed(path, payload, args, algorithm, run_tag):
    full_payload = dict(payload)
    full_payload.update(_build_mat_metadata(args, algorithm, run_tag))
    root, ext = os.path.splitext(str(path))
    seed_value = int(getattr(args, "seed", DEFAULT_SEED))
    savemat("{0}_seed{1}{2}".format(root, seed_value, ext), full_payload)


def _refresh_max_steps(args):
    args.MAX_STEPS = 2 * int(args.T) + int(args.num_update_time) * int(args.num_new_data)
    return args


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


def _normalize_old_policy_bq_list(bq_list):
    run_tags = []
    for item in bq_list:
        if not isinstance(item, (list, tuple)) or len(item) != 2:
            raise ValueError("OLD_POLICY_BQ_LIST item must be a (b, q) pair. got {0!r}".format(item))
        batch_size = _parse_positive_int(item[0], "b", item)
        q_update_time = _parse_positive_int(item[1], "q", item)
        run_tags.append(_format_old_policy_run_tag(batch_size, q_update_time))
    return _dedupe_run_tags(run_tags)


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
            raise ValueError(
                "invalid --old-policies spec {0!r}. expected format like b100:q1,b500:q10".format(spec)
            )
        batch_part, q_part = parts
        if (len(batch_part) <= 1) or (batch_part[0].lower() != "b"):
            raise ValueError(
                "invalid --old-policies spec {0!r}. expected the batch part to look like b100".format(spec)
            )
        if (len(q_part) <= 1) or (q_part[0].lower() != "q"):
            raise ValueError(
                "invalid --old-policies spec {0!r}. expected the q part to look like q1".format(spec)
            )
        batch_size = _parse_positive_int(batch_part[1:], "b", spec)
        q_update_time = _parse_positive_int(q_part[1:], "q", spec)
        run_tags.append(_format_old_policy_run_tag(batch_size, q_update_time))
    return _dedupe_run_tags(run_tags)


def _normalize_new_policy_init_spec(init_spec):
    if init_spec is None:
        return ""
    if isinstance(init_spec, str):
        run_tags = _parse_old_policy_cli(init_spec)
    elif isinstance(init_spec, (list, tuple)) and len(init_spec) == 2 and (not isinstance(init_spec[0], (list, tuple))):
        run_tags = _normalize_old_policy_bq_list([init_spec])
    else:
        raise ValueError(
            "NEW_POLICY_INIT_BQ must be None, a (b, q) pair, or a string like 'b100:q10'. got {0!r}".format(
                init_spec
            )
        )
    if len(run_tags) > 1:
        raise ValueError(
            "new policy init expects a single (b, q) pair. got {0}".format(", ".join(run_tags))
        )
    return "" if not run_tags else str(run_tags[0])


def _resolve_old_policy_args(args):
    args.seed = int(getattr(args, "seed", DEFAULT_SEED))
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
    # Fused_CPRO.py 内部沿用历史字段名，这里同步写入兼容别名。
    args.pretrain_episode = pretrain_episode
    args.checkpoint_root = checkpoint_root

    if run_tags and (pretrain_episode <= 0):
        raise ValueError(
            "old policy pretrain_episode must be a positive integer when old policies are configured. got {0}".format(
                pretrain_episode
            )
        )
    return args


def _resolve_new_policy_init_args(args):
    args.load_new_actor = bool(getattr(args, "load_new_actor", LOAD_NEW_ACTOR))
    args.new_policy_seed = int(getattr(args, "new_policy_seed", DEFAULT_SEED))

    if getattr(args, "new_policy_pretrain_episode", None) is None:
        pretrain_episode = int(NEW_POLICY_INIT_PRETRAIN_EPISODE)
    else:
        pretrain_episode = int(args.new_policy_pretrain_episode)

    checkpoint_root = getattr(args, "new_policy_checkpoint_root", None) or NEW_POLICY_INIT_CHECKPOINT_ROOT

    args.new_policy_pretrain_episode = pretrain_episode
    args.new_policy_checkpoint_root = checkpoint_root

    if not args.load_new_actor:
        args.new_policy_run_tag = ""
        return args

    init_spec = getattr(args, "new_policy_init", NEW_POLICY_INIT_BQ)
    run_tag = _normalize_new_policy_init_spec(init_spec)
    args.new_policy_run_tag = run_tag

    if not run_tag:
        raise ValueError("new actor init is enabled but new_policy_init is empty.")

    if pretrain_episode <= 0:
        raise ValueError(
            "new policy pretrain_episode must be a positive integer when new actor init is configured. got {0}".format(
                pretrain_episode
            )
        )
    return args


def _validate_old_policy_checkpoints(args):
    run_tags = [tag.strip() for tag in str(getattr(args, "old_policy_run_tags", "")).split(",") if tag.strip()]
    if not run_tags:
        return args

    resolved_paths = []
    for run_tag in run_tags:
        checkpoint_path = _resolve_sldac_checkpoint_path(
            args,
            EXAMPLE_NAME,
            run_tag,
            int(args.old_policy_pretrain_episode),
            int(args.old_policy_seed),
        )
        resolved_paths.append((run_tag, checkpoint_path))

    print("selected old policy run_tags:", ", ".join(run_tags))
    print("selected old policy seed:", int(args.old_policy_seed))
    print("selected old policy pretrain_episode:", int(args.old_policy_pretrain_episode))
    for run_tag, checkpoint_path in resolved_paths:
        print("verified old policy checkpoint:", run_tag, "->", checkpoint_path)
    return args


def _validate_new_policy_checkpoint(args):
    if not bool(getattr(args, "load_new_actor", LOAD_NEW_ACTOR)):
        return args

    run_tag = str(getattr(args, "new_policy_run_tag", "")).strip()
    if not run_tag:
        return args

    checkpoint_path = _resolve_sldac_checkpoint_path(
        args,
        EXAMPLE_NAME,
        run_tag,
        int(args.new_policy_pretrain_episode),
        int(args.new_policy_seed),
        checkpoint_root=args.new_policy_checkpoint_root,
    )
    print("selected new policy init run_tag:", run_tag)
    print("selected new policy init seed:", int(args.new_policy_seed))
    print("selected new policy init pretrain_episode:", int(args.new_policy_pretrain_episode))
    print("verified new policy init checkpoint:", run_tag, "->", checkpoint_path)
    return args


def _moving_average(values, window=5):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return arr
    out = np.zeros_like(arr)
    for idx in range(arr.size):
        left = max(0, idx - int(window) + 1)
        out[idx] = np.mean(arr[left : idx + 1])
    return out


def _coerce_split_xi_history(xi_history):
    if isinstance(xi_history, dict):
        actor_xi = np.asarray(xi_history.get("actor", []), dtype=np.float64).reshape(-1)
        critic_xi = np.asarray(xi_history.get("critic", []), dtype=np.float64).reshape(-1)
        return actor_xi, critic_xi
    actor_xi = np.asarray(xi_history, dtype=np.float64).reshape(-1)
    return actor_xi, np.zeros((0,), dtype=np.float64)


def _finalize_rho_lower_bounds(args):
    args.rho_min_new_actor = float(getattr(args, "rho_min_new_actor", DEFAULT_RHO_MIN_NEW_ACTOR))
    args.rho_min_old_policy = float(getattr(args, "rho_min_old_policy", DEFAULT_RHO_MIN_OLD_POLICY))
    if args.rho_min_new_actor < 0.0:
        raise ValueError("rho_min_new_actor must be non-negative. got {0}".format(args.rho_min_new_actor))
    if args.rho_min_old_policy < 0.0:
        raise ValueError("rho_min_old_policy must be non-negative. got {0}".format(args.rho_min_old_policy))
    run_tags = [tag.strip() for tag in str(getattr(args, "old_policy_run_tags", "")).split(",") if tag.strip()]
    rho_dim = 2 + len(run_tags)
    rho_floor_sum = float(args.rho_min_new_actor) + float(max(rho_dim - 1, 0)) * float(args.rho_min_old_policy)
    if rho_floor_sum > 1.0:
        raise ValueError(
            "rho lower bounds are infeasible for rho_dim={0}. got rho_min_new_actor={1}, rho_min_old_policy={2}, sum={3}".format(
                rho_dim,
                args.rho_min_new_actor,
                args.rho_min_old_policy,
                rho_floor_sum,
            )
        )
    return args


def _finalize_freeze_rho_args(args):
    args.freeze_rho_episode_count = int(
        getattr(args, "freeze_rho_episode_count", DEFAULT_FREEZE_RHO_EPISODE_COUNT)
    )
    if args.freeze_rho_episode_count < 0:
        raise ValueError(
            "freeze_rho_episode_count must be a non-negative integer. got {0}".format(
                args.freeze_rho_episode_count
            )
        )
    return args


def _build_artifact_name(kind, run_tag, suffix="mat"):
    return "{0}_{1}_{2}.{3}".format(ALGORITHM_NAME, str(kind), str(run_tag), str(suffix))


def _build_plot_name(kind, run_tag, seed):
    return "{0}_{1}_{2}_seed{3}.png".format(ALGORITHM_NAME, str(kind), str(run_tag), int(seed))


def _plot_reuse_probability(output_suffix, rho_history, rho_labels, xi_history, seed):
    if rho_history.size == 0:
        return
    out_path = build_algorithm_artifact_path(
        BASE_DIR,
        ALGORITHM_NAME,
        _build_plot_name("reuse_prob", output_suffix, seed),
    )
    x = np.arange(1, rho_history.shape[0] + 1)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for idx, label in enumerate(rho_labels):
        axes[0].plot(x, rho_history[:, idx], linewidth=2.0, label=label)
    axes[0].set_ylabel("Reuse probability")
    axes[0].set_title("{0} reuse probabilities: {1}".format(ALGORITHM_LABEL, output_suffix))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=2)

    actor_xi_history, critic_xi_history = _coerce_split_xi_history(xi_history)
    if actor_xi_history.size > 0:
        axes[1].plot(x, actor_xi_history, color="#222222", linewidth=2.0, label="actor_xi")
    if critic_xi_history.size > 0:
        axes[1].plot(x, critic_xi_history, color="#c23b22", linewidth=2.0, label="critic_xi")
    axes[1].set_xlabel("Episode")
    axes[1].set_ylabel("offline weight")
    axes[1].grid(alpha=0.25)
    if (actor_xi_history.size > 0) or (critic_xi_history.size > 0):
        axes[1].legend(frameon=False, ncol=2)

    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_drift_speed(output_suffix, drift_history, seed):
    update_index = np.asarray(drift_history.get("update_index", []), dtype=np.float64).reshape(-1)
    if update_index.size <= 0:
        return
    actor_rms = np.asarray(drift_history.get("actor_rms", []), dtype=np.float64).reshape(-1)
    critic_rms = np.asarray(drift_history.get("critic_rms", []), dtype=np.float64).reshape(-1)
    rho_rms = np.asarray(drift_history.get("rho_rms", []), dtype=np.float64).reshape(-1)

    out_path = build_algorithm_artifact_path(
        BASE_DIR,
        ALGORITHM_NAME,
        _build_plot_name("drift_speed", output_suffix, seed),
    )
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True)

    for values, label in ((actor_rms, "actor"), (critic_rms, "critic"), (rho_rms, "rho")):
        axes[0].plot(update_index, values, linewidth=2.0, label=label)
    axes[0].set_yscale("log")
    axes[0].set_ylabel("RMS drift")
    axes[0].set_title("{0} drift speeds: {1}".format(ALGORITHM_LABEL, output_suffix))
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, ncol=3)

    for values, label in ((actor_rms, "actor"), (critic_rms, "critic"), (rho_rms, "rho")):
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
    parser.add_argument("--beta_actor_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--beta_rho_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--xi0", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--actor-xi0", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--critic-xi0", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--xi_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--actor-xi-pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--critic-xi-pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--eta_pow", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma_pow_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--gamma_pow_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau_reward", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--tau_cost", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--cssca-solver", "--cssca_solver", dest="cssca_solver", choices=["cvx", "dual"], default=argparse.SUPPRESS)
    parser.add_argument("--rho-min-new-actor", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rho-min-old-policy", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--freeze-rho-episode-count", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--actor-distribution", dest="actor_distribution", type=str, choices=["squashed", "legacy"], default=argparse.SUPPRESS)
    parser.add_argument("--rho-beta-peak-episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--rho-beta-peak-value", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--rho-beta-end-value", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--old-policies", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--old-policy-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--old-policy-pretrain-episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--old-policy-checkpoint-root", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--new-policy-init", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--new-policy-seed", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--new-policy-pretrain-episode", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--new-policy-checkpoint-root", type=str, default=argparse.SUPPRESS)
    return parser


def _finalize_rho_new_args(args):
    args.beta_actor_pow = float(getattr(args, "beta_actor_pow", DEFAULT_BETA_ACTOR_POW))
    args.beta_rho_pow = float(getattr(args, "beta_rho_pow", DEFAULT_BETA_RHO_POW))
    args.xi0 = float(getattr(args, "xi0", DEFAULT_XI0))
    args.xi_pow = float(getattr(args, "xi_pow", DEFAULT_XI_POW))
    args.rho_beta_peak_episode = int(getattr(args, "rho_beta_peak_episode", DEFAULT_RHO_BETA_PEAK_EPISODE))
    args.rho_beta_peak_value = float(getattr(args, "rho_beta_peak_value", DEFAULT_RHO_BETA_PEAK_VALUE))
    args.rho_beta_end_value = float(getattr(args, "rho_beta_end_value", DEFAULT_RHO_BETA_END_VALUE))
    args.rho_scheduler = RHO_SCHEDULER

    if (args.xi0 < 0.0) or (args.xi0 > 1.0):
        raise ValueError("xi0 must be in [0, 1] as offline weight. got xi0={0}".format(args.xi0))
    if args.xi_pow <= 0.0:
        raise ValueError("xi_pow must be positive. got xi_pow={0}".format(args.xi_pow))
    if getattr(args, "actor_xi0", None) is None:
        args.actor_xi0 = float(args.xi0)
    else:
        args.actor_xi0 = float(args.actor_xi0)
    if getattr(args, "critic_xi0", None) is None:
        args.critic_xi0 = float(args.xi0)
    else:
        args.critic_xi0 = float(args.critic_xi0)
    if getattr(args, "actor_xi_pow", None) is None:
        args.actor_xi_pow = float(args.xi_pow)
    else:
        args.actor_xi_pow = float(args.actor_xi_pow)
    if getattr(args, "critic_xi_pow", None) is None:
        args.critic_xi_pow = float(args.xi_pow)
    else:
        args.critic_xi_pow = float(args.critic_xi_pow)
    for name in ("actor_xi0", "critic_xi0"):
        value = float(getattr(args, name))
        if (value < 0.0) or (value > 1.0):
            raise ValueError("{0} must be in [0, 1] as offline weight. got {0}={1}".format(name, value))
    for name in ("actor_xi_pow", "critic_xi_pow"):
        value = float(getattr(args, name))
        if value <= 0.0:
            raise ValueError("{0} must be positive. got {0}={1}".format(name, value))
    if args.rho_beta_peak_episode <= 0:
        raise ValueError(
            "rho_beta_peak_episode must be a positive integer. got rho_beta_peak_episode={0}".format(
                args.rho_beta_peak_episode
            )
        )
    if (args.rho_beta_peak_value <= 0.0) or (args.rho_beta_peak_value > 1.0):
        raise ValueError(
            "rho_beta_peak_value must be in (0, 1]. got rho_beta_peak_value={0}".format(args.rho_beta_peak_value)
        )
    if args.rho_beta_end_value <= 0.0:
        raise ValueError("rho_beta_end_value must be positive. got rho_beta_end_value={0}".format(args.rho_beta_end_value))
    if args.rho_beta_end_value >= args.rho_beta_peak_value:
        raise ValueError(
            "rho_beta_end_value must be smaller than rho_beta_peak_value. got rho_beta_end_value={0}, rho_beta_peak_value={1}".format(
                args.rho_beta_end_value,
                args.rho_beta_peak_value,
            )
        )
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
    for output_suffix, message, t_horizon, grad_t, num_new_data_run, q_update_time in FUSED_CPRO_RHONEW_RUNS:
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
        reward_save, cost_save, rho_history, xi_history, rho_labels, drift_history = Fused_CPRO_RhoNew_main(
            run_args,
            EXAMPLE_NAME,
        )
        actor_xi_history, critic_xi_history = _coerce_split_xi_history(xi_history)
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, _build_artifact_name("reward", output_suffix)),
            {"array": reward_save},
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, _build_artifact_name("cost", output_suffix)),
            {"array": cost_save},
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, _build_artifact_name("rho", output_suffix)),
            {
                "array": rho_history,
                "labels": np.asarray(rho_labels, dtype="U32"),
                "xi": actor_xi_history,
                "actor_xi": actor_xi_history,
                "critic_xi": critic_xi_history,
            },
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )
        _save_mat_with_seed(
            build_algorithm_artifact_path(BASE_DIR, ALGORITHM_NAME, _build_artifact_name("drift", output_suffix)),
            drift_history,
            run_args,
            ALGORITHM_NAME,
            output_suffix,
        )
        _plot_reuse_probability(output_suffix, rho_history, rho_labels, xi_history, run_args.seed)
        _plot_drift_speed(output_suffix, drift_history, run_args.seed)


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
    args = _finalize_rho_new_args(args)
    args = _resolve_old_policy_args(args)
    args = _resolve_new_policy_init_args(args)
    args = _finalize_rho_lower_bounds(args)
    args = _finalize_freeze_rho_args(args)
    _migrate_legacy_checkpoints(
        args.old_policy_checkpoint_root,
        EXAMPLE_NAME,
        default_seed=int(args.old_policy_seed),
    )
    args = _validate_old_policy_checkpoints(args)
    args = _validate_new_policy_checkpoint(args)
    experiment_seeds = resolve_experiment_seeds(args, DEFAULT_SEED)
    print("experiment seeds:", ", ".join(str(seed_value) for seed_value in experiment_seeds))
    for seed_value in experiment_seeds:
        print("run seed:", int(seed_value))
        seed_args = argparse.Namespace(**vars(args))
        seed_args.seed = int(seed_value)
        _run_single_seed(seed_args)


if __name__ == "__main__":
    main()
