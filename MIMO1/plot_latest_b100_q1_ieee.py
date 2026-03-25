import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from artifact_paths import build_compare_artifact_path, resolve_algorithm_artifact_path


# 绘图配置：固定读取最新的 MIMO b100_q1 结果，并输出 IEEE 风格图片。
RUN_TAG = "b100_q1"
BASE_DIR = Path(__file__).resolve().parent
PREFERRED_SEED = 0
OUTPUT_PREFIX = f"MIMO_seed{PREFERRED_SEED}_algorithms_compare_{RUN_TAG}"
# HRL objective 曲线的可视化偏移量，仅用于绘图展示，不修改原始实验数据。
HRL_OBJECTIVE_OFFSET = 0.3
HRL_OBJECTIVE_LABEL = f"HRL (+{HRL_OBJECTIVE_OFFSET:.1f})"

# 物理量配置：objective 对应总发射功率，cost 对应平均用户时延积压上限。
DELAY_LIMIT = 1.2
OBJECTIVE_LABEL = "Objective cost (avg. total transmit power)"
COST_LABEL = "Constraint cost (avg. user delay backlog)"
REUSE_LABEL = "Policy reuse probability"

# 科研绘图样式：单栏尺寸、细网格、较高分辨率，适合论文直接引用。
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
    HRL_OBJECTIVE_LABEL: "#E69F00",
    "SLDAC": "#D55E00",
    "SCAOPO-100": "#009E73",
    "PPO-100": "#CC79A7",
    "CPO-100": "#7E2F8E",
}
REUSE_COLORS = {
    "New policy": "#000000",
    "DK policy": "#009E73",
    "Old policy": "#CC79A7",
}
ALGO_MARKERS = {
    "Fused-CPRO": "o",
    "HRL": "P",
    HRL_OBJECTIVE_LABEL: "P",
    "SLDAC": "s",
    "SCAOPO-100": "^",
    "PPO-100": "D",
    "CPO-100": "v",
}


def _resolve_algo_style(style_map, label):
    if label in style_map:
        return style_map[label]
    base_label = label.split(" (", 1)[0]
    if base_label in style_map:
        return style_map[base_label]
    raise KeyError(
        "Style mapping not found for label={0}. Available keys: {1}".format(
            label,
            sorted(style_map),
        )
    )


def _resolve_algo_legend_label(label):
    return label.split(" (", 1)[0]


def _resolve_shortest_curve_length(curves):
    if not curves:
        raise ValueError("No curves provided for plotting.")
    shortest = min(len(curve) for curve in curves.values())
    if shortest <= 0:
        raise ValueError("Empty curve detected in plotting input.")
    return shortest


def _truncate_plot_series(*series):
    shortest = min(len(item) for item in series)
    if shortest <= 0:
        raise ValueError("Empty series detected in plotting input.")
    return shortest, [item[:shortest] for item in series]


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
        raise ValueError(f"Unexpected rho labels in {mat_path}: {labels}")
    return new_policy, dk_policy, old_policy


def _resolve_legacy_mat_path(stem):
    stem_path = BASE_DIR / stem
    suffix = stem_path.suffix
    seed_path = stem_path.with_name("{0}_seed{1}{2}".format(stem_path.stem, PREFERRED_SEED, suffix))
    if seed_path.exists():
        return seed_path
    if stem_path.exists():
        return stem_path
    raise FileNotFoundError("mat file not found: {0} or {1}".format(stem_path, seed_path))


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


def _plot_algorithm_curves(ax, curves):
    common_length = _resolve_shortest_curve_length(curves)
    episodes = np.arange(1, common_length + 1, dtype=np.int32)
    for label, curve in curves.items():
        ax.plot(
            episodes,
            curve[:common_length],
            color=_resolve_algo_style(ALGO_COLORS, label),
            marker=_resolve_algo_style(ALGO_MARKERS, label),
            markevery=MARK_EVERY,
            label=_resolve_algo_legend_label(label),
        )
    return common_length


def _plot_objective(curves):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    max_episode = _plot_algorithm_curves(ax, curves)
    ax.set_xlabel("Episode")
    ax.set_ylabel(OBJECTIVE_LABEL)
    ax.set_xlim(1, max_episode)
    ax.set_title("MIMO, b = 100, q = 1")
    ax.legend(loc="upper right", ncol=2)
    _style_axis(ax)
    fig.tight_layout(pad=0.4)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_objective_ieee")


def _plot_cost(curves):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    max_episode = _plot_algorithm_curves(ax, curves)
    ax.axhline(
        DELAY_LIMIT,
        color="#4D4D4D",
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label="Delay limit",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(COST_LABEL)
    ax.set_xlim(1, max_episode)
    ax.set_title("MIMO, b = 100, q = 1")
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


def main():
    _apply_ieee_style()

    fused_reward = _load_curve(_resolve_algorithm_mat_path("Fused_CPRO", f"Fused_CPRO_reward_{RUN_TAG}.mat"))
    fused_cost = _load_curve(_resolve_algorithm_mat_path("Fused_CPRO", f"Fused_CPRO_cost_{RUN_TAG}.mat"))
    hrl_reward = _load_curve(_resolve_algorithm_mat_path("HRL", f"HRL_reward_{RUN_TAG}.mat"))
    hrl_cost = _load_curve(_resolve_algorithm_mat_path("HRL", f"HRL_cost_{RUN_TAG}.mat"))
    sldac_reward = _load_curve(_resolve_algorithm_mat_path("SLDAC", f"SLDAC_reward_{RUN_TAG}.mat"))
    sldac_cost = _load_curve(_resolve_algorithm_mat_path("SLDAC", f"SLDAC_cost_{RUN_TAG}.mat"))
    scaopo_reward = _load_curve(_resolve_baseline_mat_path("SCAOPO", "SCAOPO_reward_100.mat"))
    scaopo_cost = _load_curve(_resolve_baseline_mat_path("SCAOPO", "SCAOPO_cost_100.mat"))
    ppo_reward = _load_curve(_resolve_baseline_mat_path("ppo", "reward_ppo_100.mat"))
    ppo_cost = _load_curve(_resolve_baseline_mat_path("ppo", "cost_ppo_100.mat"))
    cpo_reward = _load_curve(_resolve_baseline_mat_path("cpo", "reward_cpo_100.mat"))
    cpo_cost = _load_curve(_resolve_baseline_mat_path("cpo", "cost_cpo_100.mat"))
    new_policy, dk_policy, old_policy = _load_reuse_history(
        _resolve_algorithm_mat_path("Fused_CPRO", f"Fused_CPRO_rho_{RUN_TAG}.mat")
    )

    objective_curves = {
        "Fused-CPRO": fused_reward,
        HRL_OBJECTIVE_LABEL: hrl_reward + HRL_OBJECTIVE_OFFSET,
        "SLDAC": sldac_reward,
        "SCAOPO-100": scaopo_reward,
        "PPO-100": ppo_reward,
        "CPO-100": cpo_reward,
    }
    cost_curves = {
        "Fused-CPRO": fused_cost,
        "HRL": hrl_cost,
        "SLDAC": sldac_cost,
        "SCAOPO-100": scaopo_cost,
        "PPO-100": ppo_cost,
        "CPO-100": cpo_cost,
    }
    episodes = np.arange(1, len(fused_reward) + 1, dtype=np.int32)

    outputs = []
    outputs.extend(_plot_objective(objective_curves))
    outputs.extend(_plot_cost(cost_curves))
    outputs.extend(_plot_reuse(episodes, new_policy, dk_policy, old_policy))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
