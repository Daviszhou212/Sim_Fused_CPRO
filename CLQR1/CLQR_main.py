import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from artifact_paths import build_compare_artifact_path, resolve_algorithm_artifact_path


# 绘图配置：固定读取 CLQR1 当前主线结果，输出 IEEE 风格图片。
BASE_DIR = Path(__file__).resolve().parent
PREFERRED_SEED = 0
FUSED_RUN_TAG = "default"
OUTPUT_PREFIX = f"CLQR_seed{PREFERRED_SEED}_algorithms_compare"

# 物理量配置：objective 表示待最小化的平均目标代价，cost 表示约束代价。
CONSTRAINT_LIMIT = 380.0
OBJECTIVE_LABEL = "Objective cost (avg. quadratic objective)"
COST_LABEL = "Constraint cost (avg. quadratic constraint)"
REUSE_LABEL = "Policy reuse probability"

# 科研绘图样式：与 MIMO1 保持一致的 IEEE 风格。
FIG_WIDTH = 3.55
FIG_HEIGHT = 2.65
FIG_DPI = 600
LINE_WIDTH = 1.45
REFERENCE_LINE_WIDTH = 1.15
MARKER_SIZE = 5.0
MARK_EVERY = 6

ALGO_COLORS = {
    "Fused-CPRO": "#0072B2",
    "HRL": "#E69F00",
    "SLDAC-no reuse": "#009E73",
    "SLDAC-q1": "#56B4E9",
    "SLDAC-q5": "#CC79A7",
    "SLDAC-q10": "#D55E00",
    "ACPO": "#4D4D4D",
    "SCAOPO-500": "#64B5CD",
    "PPO-100": "#8172B2",
    "CPO-100": "#7E2F8E",
}
ALGO_MARKERS = {
    "Fused-CPRO": "o",
    "HRL": "P",
    "SLDAC-no reuse": "^",
    "SLDAC-q1": "s",
    "SLDAC-q5": "D",
    "SLDAC-q10": "X",
    "ACPO": "*",
    "SCAOPO-500": "v",
    "PPO-100": "<",
    "CPO-100": ">",
}
REUSE_COLORS = {
    "New policy": "#000000",
    "DK policy": "#009E73",
    "Old policy": "#CC79A7",
}


def _apply_ieee_style():
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "axes.labelsize": 9,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.linewidth": 0.9,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.6,
            "grid.alpha": 0.25,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _load_curve(mat_path):
    data = loadmat(mat_path)
    return np.asarray(data["array"], dtype=np.float64).reshape(-1)


def _load_reuse_history(mat_path):
    data = loadmat(mat_path)
    rho = np.asarray(data["array"], dtype=np.float64)
    raw_labels = np.asarray(data["labels"]).reshape(-1)
    labels = [str(item).strip() for item in raw_labels]

    new_policy = None
    dk_policy = None
    old_policy = np.zeros((rho.shape[0],), dtype=np.float64)

    for idx, label in enumerate(labels):
        label_lower = label.lower()
        if label_lower == "new_actor":
            new_policy = rho[:, idx]
        elif label_lower == "dk_policy":
            dk_policy = rho[:, idx]
        else:
            old_policy += rho[:, idx]

    if new_policy is None or dk_policy is None:
        raise ValueError("Unexpected rho labels in {0}: {1}".format(mat_path, labels))
    return new_policy, dk_policy, old_policy


def _resolve_algorithm_mat_path(algorithm_name, stem):
    stem_path = Path(stem)
    candidates = [
        "{0}_seed{1}{2}".format(stem_path.stem, PREFERRED_SEED, stem_path.suffix),
        str(stem),
    ]
    for candidate in candidates:
        try:
            return Path(resolve_algorithm_artifact_path(str(BASE_DIR), algorithm_name, candidate))
        except FileNotFoundError:
            continue
    raise FileNotFoundError(
        "mat file not found for algorithm={0}: {1}".format(
            algorithm_name,
            stem,
        )
    )


def _resolve_baseline_mat_path(algorithm_name, stem):
    return Path(resolve_algorithm_artifact_path(str(BASE_DIR), algorithm_name, stem))


def _try_load_curve(path_getter):
    try:
        return _load_curve(path_getter())
    except FileNotFoundError:
        return None


def _try_load_reuse_history(path_getter):
    try:
        return _load_reuse_history(path_getter())
    except FileNotFoundError:
        return None


def _style_axis(ax):
    ax.grid(True, axis="y")
    ax.grid(True, axis="x", alpha=0.12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _save_figure(fig, stem):
    png_path = Path(build_compare_artifact_path(str(BASE_DIR), f"{stem}.png"))
    pdf_path = Path(build_compare_artifact_path(str(BASE_DIR), f"{stem}.pdf"))
    fig.savefig(png_path, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return png_path, pdf_path


def _resolve_shortest_curve_length(curves):
    if not curves:
        raise ValueError("No curves provided for plotting.")
    shortest = min(len(values) for _, values in curves)
    if shortest <= 0:
        raise ValueError("Empty curve detected in plotting input.")
    return shortest


def _truncate_plot_series(*series):
    shortest = min(len(item) for item in series)
    if shortest <= 0:
        raise ValueError("Empty series detected in plotting input.")
    return shortest, [item[:shortest] for item in series]


def _plot_algorithm_curves(ax, curves):
    common_length = _resolve_shortest_curve_length(curves)
    episodes = np.arange(1, common_length + 1, dtype=np.int32)
    for label, values in curves:
        ax.plot(
            episodes,
            values[:common_length],
            color=ALGO_COLORS[label],
            marker=ALGO_MARKERS[label],
            markevery=MARK_EVERY,
            label=label,
        )
    return common_length


def _plot_objective(curves):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    max_episode = _plot_algorithm_curves(ax, curves)
    ax.set_xlabel("Episode")
    ax.set_ylabel(OBJECTIVE_LABEL)
    ax.set_xlim(1, max_episode)
    ax.set_title("CLQR")
    ax.legend(loc="upper right", ncol=2)
    _style_axis(ax)
    fig.tight_layout(pad=0.4)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_objective_ieee")


def _plot_cost(curves):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    max_episode = _plot_algorithm_curves(ax, curves)
    ax.axhline(
        CONSTRAINT_LIMIT,
        color="#4D4D4D",
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label="Constraint limit",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(COST_LABEL)
    ax.set_xlim(1, max_episode)
    ax.set_title("CLQR")
    ax.legend(loc="upper right", ncol=2)
    _style_axis(ax)
    fig.tight_layout(pad=0.4)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_cost_ieee")


def _plot_reuse(episodes, new_policy, dk_policy, old_policy):
    common_length, (episodes, new_policy, dk_policy, old_policy) = _truncate_plot_series(
        episodes,
        new_policy,
        dk_policy,
        old_policy,
    )
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    ax.plot(
        episodes,
        new_policy,
        color=REUSE_COLORS["New policy"],
        marker="o",
        markevery=MARK_EVERY,
        label="New policy",
    )
    ax.plot(
        episodes,
        dk_policy,
        color=REUSE_COLORS["DK policy"],
        marker="^",
        markevery=MARK_EVERY,
        label="DK policy",
    )
    ax.plot(
        episodes,
        old_policy,
        color=REUSE_COLORS["Old policy"],
        marker="s",
        markevery=MARK_EVERY,
        label="Old policy",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(REUSE_LABEL)
    ax.set_xlim(1, common_length)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Fused-CPRO policy mixing weights")
    ax.legend(loc="center right")
    _style_axis(ax)
    fig.tight_layout(pad=0.4)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_reuse_ieee")


def _append_curve_if_present(curves, label, values):
    if values is not None:
        curves.append((label, values))


def main():
    _apply_ieee_style()

    objective_curves = []
    cost_curves = []

    fused_reward = _try_load_curve(
        lambda: _resolve_algorithm_mat_path("Fused_CPRO", f"Fused_CPRO_reward_{FUSED_RUN_TAG}.mat")
    )
    fused_cost = _try_load_curve(
        lambda: _resolve_algorithm_mat_path("Fused_CPRO", f"Fused_CPRO_cost_{FUSED_RUN_TAG}.mat")
    )
    _append_curve_if_present(objective_curves, "Fused-CPRO", fused_reward)
    _append_curve_if_present(cost_curves, "Fused-CPRO", fused_cost)

    hrl_reward = _try_load_curve(lambda: _resolve_algorithm_mat_path("HRL", f"HRL_reward_{FUSED_RUN_TAG}.mat"))
    hrl_cost = _try_load_curve(lambda: _resolve_algorithm_mat_path("HRL", f"HRL_cost_{FUSED_RUN_TAG}.mat"))
    _append_curve_if_present(objective_curves, "HRL", hrl_reward)
    _append_curve_if_present(cost_curves, "HRL", hrl_cost)

    sldac_specs = [
        ("SLDAC-q1", "b100_q1"),
    ]
    for label, run_tag in sldac_specs:
        reward_curve = _try_load_curve(
            lambda run_tag=run_tag: _resolve_algorithm_mat_path("SLDAC", f"SLDAC_reward_{run_tag}.mat")
        )
        cost_curve = _try_load_curve(
            lambda run_tag=run_tag: _resolve_algorithm_mat_path("SLDAC", f"SLDAC_cost_{run_tag}.mat")
        )
        if reward_curve is None or cost_curve is None:
            continue
        objective_curves.append((label, reward_curve))
        cost_curves.append((label, cost_curve))

    acpo_reward = _try_load_curve(lambda: _resolve_algorithm_mat_path("ACPO", "ACPO_reward_b250.mat"))
    acpo_cost = _try_load_curve(lambda: _resolve_algorithm_mat_path("ACPO", "ACPO_cost_b250.mat"))
    _append_curve_if_present(objective_curves, "ACPO", acpo_reward)
    _append_curve_if_present(cost_curves, "ACPO", acpo_cost)

    scaopo_reward = _try_load_curve(lambda: _resolve_baseline_mat_path("SCAOPO", "SCAOPO_reward_500.mat"))
    scaopo_cost = _try_load_curve(lambda: _resolve_baseline_mat_path("SCAOPO", "SCAOPO_cost_500.mat"))
    _append_curve_if_present(objective_curves, "SCAOPO-500", scaopo_reward)
    _append_curve_if_present(cost_curves, "SCAOPO-500", scaopo_cost)

    ppo_reward = _try_load_curve(lambda: _resolve_baseline_mat_path("ppo", "reward_ppo_100.mat"))
    ppo_cost = _try_load_curve(lambda: _resolve_baseline_mat_path("ppo", "cost_ppo_100.mat"))
    _append_curve_if_present(objective_curves, "PPO-100", ppo_reward)
    _append_curve_if_present(cost_curves, "PPO-100", ppo_cost)

    cpo_reward = _try_load_curve(lambda: _resolve_baseline_mat_path("cpo", "reward_cpo_100.mat"))
    cpo_cost = _try_load_curve(lambda: _resolve_baseline_mat_path("cpo", "cost_cpo_100.mat"))
    _append_curve_if_present(objective_curves, "CPO-100", cpo_reward)
    _append_curve_if_present(cost_curves, "CPO-100", cpo_cost)

    if not objective_curves or not cost_curves:
        raise FileNotFoundError("no CLQR curves were found in outputs or legacy root.")

    outputs = []
    outputs.extend(_plot_objective(objective_curves))
    outputs.extend(_plot_cost(cost_curves))

    reuse_data = _try_load_reuse_history(
        lambda: _resolve_algorithm_mat_path("Fused_CPRO", f"Fused_CPRO_rho_{FUSED_RUN_TAG}.mat")
    )
    if reuse_data is not None and fused_reward is not None:
        episodes = np.arange(1, len(fused_reward) + 1, dtype=np.int32)
        outputs.extend(_plot_reuse(episodes, *reuse_data))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
