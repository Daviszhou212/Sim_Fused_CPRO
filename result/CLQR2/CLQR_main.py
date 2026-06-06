import re
import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from artifact_paths import build_compare_artifact_path, resolve_algorithm_artifact_path
from plot_series_styles import apply_series_style

# 绘图结果配置：优先读取带指定 seed 后缀的 mat 文件，并控制输出文件名前缀。
PREFERRED_SEED = 0
OUTPUT_PREFIX = f"CLQR_seed{PREFERRED_SEED}_algorithms_compare"

# 物理量配置：objective 表示待最小化的平均目标代价，cost 表示约束代价。
CONSTRAINT_LIMIT = 380.0
OBJECTIVE_LABEL = "Objective cost (avg. quadratic objective)"
COST_LABEL = "Constraint cost (avg. quadratic constraint)"
REUSE_LABEL = "Policy reuse probability"

# 平滑显示配置：仅影响画图展示，不修改原始 mat 数据。
FUSED_CPRO_SMOOTH_ENABLE = False
FUSED_CPRO_SMOOTH_WINDOW = 5
FUSED_CPRO_COS_RHO_SMOOTH_ENABLE = False
FUSED_CPRO_COS_RHO_SMOOTH_WINDOW = 5
HRL_SMOOTH_ENABLE = False
HRL_SMOOTH_WINDOW = 5
PRCRL_SMOOTH_ENABLE = False
PRCRL_SMOOTH_WINDOW = 5
SLDAC_SMOOTH_ENABLE = True
SLDAC_SMOOTH_WINDOW = 15
DK_SMOOTH_ENABLE = False
DK_SMOOTH_WINDOW = 5
ACPO_SMOOTH_ENABLE = False
ACPO_SMOOTH_WINDOW = 5
SCAOPO_SMOOTH_ENABLE = False
SCAOPO_SMOOTH_WINDOW = 5
PPO_SMOOTH_ENABLE = False
PPO_SMOOTH_WINDOW = 5
CPO_SMOOTH_ENABLE = False
CPO_SMOOTH_WINDOW = 5

# 绘图目标配置：直接在这里增删曲线即可，不依赖 CLI。
PLOT_SERIES = [
    {
        "label": "Fused-CPRO",
        "artifact_group": "Fused_CPRO",
        "reward_stem": "Fused_CPRO_reward_default.mat",
        "cost_stem": "Fused_CPRO_cost_default.mat",
        "rho_stem": "Fused_CPRO_rho_default.mat",
        "prefer_seed_suffix": True,
        "smooth_enable": FUSED_CPRO_SMOOTH_ENABLE,
        "smooth_window": FUSED_CPRO_SMOOTH_WINDOW,
    },
    # {
    #     "label": "Fused-CPRO-CosRho",
    #     "artifact_group": "Fused_CPRO_CosRho",
    #     "reward_stem": "Fused_CPRO_CosRho_reward_default.mat",
    #     "cost_stem": "Fused_CPRO_CosRho_cost_default.mat",
    #     "rho_stem": "Fused_CPRO_CosRho_rho_default.mat",
    #     "prefer_seed_suffix": True,
    #     "smooth_enable": FUSED_CPRO_COS_RHO_SMOOTH_ENABLE,
    #     "smooth_window": FUSED_CPRO_COS_RHO_SMOOTH_WINDOW,
    # },
    # {
    #     "label": "HRL",
    #     "artifact_group": "HRL",
    #     "reward_stem": "HRL_reward_default.mat",
    #     "cost_stem": "HRL_cost_default.mat",
    #     "rho_stem": "HRL_rho_default.mat",
    #     "prefer_seed_suffix": True,
    #     "smooth_enable": HRL_SMOOTH_ENABLE,
    #     "smooth_window": HRL_SMOOTH_WINDOW,
    # },
    {
        "label": "PRCRL",
        "artifact_group": "PRCRL",
        "reward_stem": "PRCRL_reward_default.mat",
        "cost_stem": "PRCRL_cost_default.mat",
        "rho_stem": "PRCRL_rho_default.mat",
        "prefer_seed_suffix": True,
        "smooth_enable": PRCRL_SMOOTH_ENABLE,
        "smooth_window": PRCRL_SMOOTH_WINDOW,
    },
    {
        "label": "SLDAC",
        "artifact_group": "SLDAC",
        "reward_stem": "SLDAC_reward_b100_q5.mat",
        "cost_stem": "SLDAC_cost_b100_q5.mat",
        "prefer_seed_suffix": True,
        "smooth_enable": SLDAC_SMOOTH_ENABLE,
        "smooth_window": SLDAC_SMOOTH_WINDOW,
    },
    # {
    #     "label": "DK",
    #     "artifact_group": "DK",
    #     "reward_stem": "DK_reward_default.mat",
    #     "cost_stem": "DK_cost_default.mat",
    #     "prefer_seed_suffix": True,
    #     "smooth_enable": DK_SMOOTH_ENABLE,
    #     "smooth_window": DK_SMOOTH_WINDOW,
    # },
    # {
    #     "label": "ACPO",
    #     "artifact_group": "ACPO",
    #     "reward_stem": "ACPO_reward_b500.mat",
    #     "cost_stem": "ACPO_cost_b500.mat",
    #     "prefer_seed_suffix": True,
    #     "smooth_enable": ACPO_SMOOTH_ENABLE,
    #     "smooth_window": ACPO_SMOOTH_WINDOW,
    # },
    {
        "label": "SCAOPO",
        "artifact_group": "SCAOPO",
        "reward_stem": "SCAOPO_reward_100.mat",
        "cost_stem": "SCAOPO_cost_100.mat",
        "prefer_seed_suffix": False,
        "smooth_enable": SCAOPO_SMOOTH_ENABLE,
        "smooth_window": SCAOPO_SMOOTH_WINDOW,
    },
    {
        "label": "PPO-Lag",
        "artifact_group": "ppo",
        "reward_stem": "reward_ppo_100.mat",
        "cost_stem": "cost_ppo_100.mat",
        "prefer_seed_suffix": False,
        "smooth_enable": PPO_SMOOTH_ENABLE,
        "smooth_window": PPO_SMOOTH_WINDOW,
    },
    {
        "label": "CPO",
        "artifact_group": "cpo",
        "reward_stem": "reward_cpo_100.mat",
        "cost_stem": "cost_cpo_100.mat",
        "prefer_seed_suffix": False,
        "smooth_enable": CPO_SMOOTH_ENABLE,
        "smooth_window": CPO_SMOOTH_WINDOW,
    },
]

# 统一图例文本、颜色与 marker，避免 MIMO / CLQR 漂移。
PLOT_SERIES = [apply_series_style(series_config) for series_config in PLOT_SERIES]

# 科研绘图样式：与 MIMO1 保持一致的 IEEE 风格。
FIG_WIDTH = 4.6
FIG_HEIGHT = 3.35
FIG_DPI = 600
LINE_WIDTH = 1.45
REFERENCE_LINE_WIDTH = 1.15
REFERENCE_LINE_COLOR = "#666666"
MARKER_SIZE = 5.0
MARK_EVERY = 6
COMPARE_LEGEND_NCOL = 4
REUSE_LEGEND_NCOL = 3
COMPARE_HEADER_RECT = (0.0, 0.0, 1.0, 0.90)
REUSE_HEADER_RECT = (0.0, 0.0, 1.0, 0.95)
COMPARE_LEGEND_Y = 0.995
REUSE_LEGEND_Y = 0.992
REUSE_PANEL_HEIGHT = 2.6
# objective 图 y 轴范围：忽略显著过高的算法，展开主流算法的下方趋势。
OBJECTIVE_YLIM_IGNORE_LABELS = {"PPO-Lag"}
OBJECTIVE_YLIM_PAD_RATIO = 0.08
OBJECTIVE_YLIM_MIN_PAD = 0.8

REUSE_COLORS = {
    "New policy": "#000000",
    "DK policy": "#009E73",
    "Old policy": "#CC79A7",
}

SERIES_REQUIRED_KEYS = (
    "label",
    "artifact_group",
    "reward_stem",
    "cost_stem",
    "color",
    "marker",
    "prefer_seed_suffix",
)
SERIES_OPTIONAL_KEYS = ("rho_stem", "smooth_enable", "smooth_window")
SEED_SUFFIX_PATTERN = re.compile(r"_seed\d+$")


def _apply_ieee_style():
    plt.rcParams.update(
        {
            "figure.dpi": FIG_DPI,
            "savefig.dpi": FIG_DPI,
            "savefig.pad_inches": 0.02,
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
            "axes.axisbelow": True,
            "lines.linewidth": LINE_WIDTH,
            "lines.markersize": MARKER_SIZE,
            "lines.solid_capstyle": "round",
            "lines.dash_capstyle": "round",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.18,
            "grid.linestyle": "--",
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.handletextpad": 0.45,
            "legend.columnspacing": 0.9,
            "legend.labelspacing": 0.25,
            "legend.borderaxespad": 0.15,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def _series_label(series_config):
    return str(series_config["label"]).strip()


def _load_curve(mat_path):
    data = loadmat(mat_path)
    return np.asarray(data["array"], dtype=np.float64).reshape(-1)


def _moving_average(values, window):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        return arr

    window = int(window)
    if window <= 1:
        return arr.copy()

    out = np.zeros_like(arr)
    for idx in range(arr.size):
        left = max(0, idx - window + 1)
        out[idx] = np.mean(arr[left : idx + 1])
    return out


def _series_smooth_enabled(series_config):
    return bool(series_config.get("smooth_enable", False))


def _series_smooth_window(series_config):
    return int(series_config.get("smooth_window", 5))


def _maybe_smooth(series_config, values):
    if not _series_smooth_enabled(series_config):
        return np.asarray(values, dtype=np.float64)
    return _moving_average(values, _series_smooth_window(series_config))


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


def _build_seed_preferred_stem(stem):
    stem_path = Path(str(stem))
    if SEED_SUFFIX_PATTERN.search(stem_path.stem):
        return str(stem_path)
    return "{0}_seed{1}{2}".format(stem_path.stem, PREFERRED_SEED, stem_path.suffix)


def _build_legacy_acpo_stem(stem):
    stem_text = str(stem)
    if "_b500" not in stem_text:
        return None
    return stem_text.replace("_b500", "_b250", 1)


def _validate_plot_series_config(plot_series):
    if not plot_series:
        raise ValueError("PLOT_SERIES must contain at least one plotting series.")

    labels = set()
    reuse_series = []

    for idx, series_config in enumerate(plot_series):
        if not isinstance(series_config, dict):
            raise TypeError("PLOT_SERIES[{0}] must be a dict.".format(idx))

        missing_keys = [key for key in SERIES_REQUIRED_KEYS if key not in series_config]
        if missing_keys:
            raise KeyError(
                "PLOT_SERIES[{0}] is missing keys: {1}".format(idx, ", ".join(missing_keys))
            )

        unknown_keys = sorted(set(series_config.keys()) - set(SERIES_REQUIRED_KEYS) - set(SERIES_OPTIONAL_KEYS))
        if unknown_keys:
            raise KeyError(
                "PLOT_SERIES[{0}] contains unsupported keys: {1}".format(idx, ", ".join(unknown_keys))
            )

        label = _series_label(series_config)
        if not label:
            raise ValueError("PLOT_SERIES[{0}] label must be non-empty.".format(idx))
        if label in labels:
            raise ValueError("Duplicate plot label detected in PLOT_SERIES: {0}".format(label))
        labels.add(label)

        for key in ("artifact_group", "color", "marker"):
            if not str(series_config[key]).strip():
                raise ValueError("PLOT_SERIES[{0}] field {1} must be non-empty.".format(idx, key))

        for key in ("reward_stem", "cost_stem"):
            if Path(str(series_config[key])).suffix.lower() != ".mat":
                raise ValueError("PLOT_SERIES[{0}] field {1} must point to a .mat file.".format(idx, key))

        if not isinstance(series_config["prefer_seed_suffix"], bool):
            raise TypeError("PLOT_SERIES[{0}] field prefer_seed_suffix must be bool.".format(idx))

        rho_stem = series_config.get("rho_stem")
        if rho_stem is not None:
            if Path(str(rho_stem)).suffix.lower() != ".mat":
                raise ValueError("PLOT_SERIES[{0}] field rho_stem must point to a .mat file.".format(idx))
            if str(series_config["artifact_group"]).strip() not in (
                "Fused_CPRO",
                "Fused_CPRO_RhoNew",
                "Fused_CPRO_CosRho",
                "HRL",
                "PRCRL",
            ):
                raise ValueError(
                    "Only Fused_CPRO, Fused_CPRO_RhoNew, Fused_CPRO_CosRho, HRL or PRCRL series may define rho_stem. Invalid label: {0}".format(
                        label
                    )
                )
            reuse_series.append(series_config)

        smooth_enable = series_config.get("smooth_enable")
        if smooth_enable is not None and not isinstance(smooth_enable, bool):
            raise TypeError("PLOT_SERIES[{0}] field smooth_enable must be bool.".format(idx))

        smooth_window = series_config.get("smooth_window")
        if smooth_window is not None:
            try:
                smooth_window = int(smooth_window)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "PLOT_SERIES[{0}] field smooth_window must be an integer.".format(idx)
                ) from exc
            if smooth_window <= 0:
                raise ValueError(
                    "PLOT_SERIES[{0}] field smooth_window must be positive.".format(idx)
                )
    return reuse_series


def _resolve_series_mat_path(series_config, stem_key):
    artifact_group = str(series_config["artifact_group"])
    stem = str(series_config[stem_key])
    candidates = []
    if bool(series_config["prefer_seed_suffix"]):
        candidates.append(_build_seed_preferred_stem(stem))
    candidates.append(stem)
    if artifact_group == "ACPO":
        legacy_stem = _build_legacy_acpo_stem(stem)
        if legacy_stem is not None:
            if bool(series_config["prefer_seed_suffix"]):
                candidates.append(_build_seed_preferred_stem(legacy_stem))
            candidates.append(legacy_stem)

    last_error = None
    for candidate in candidates:
        try:
            return Path(resolve_algorithm_artifact_path(str(BASE_DIR), artifact_group, candidate))
        except FileNotFoundError as exc:
            last_error = exc
            continue

    label = _series_label(series_config)
    raise FileNotFoundError(
        "artifact not found for plot series {0!r}, field {1}: {2}".format(label, stem_key, stem)
    ) from last_error


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
    ax.grid(True, axis="x", alpha=0.08)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _resolve_objective_ylim(curves):
    target_curves = [
        (series_config, values)
        for series_config, values in curves
        if _series_label(series_config) not in OBJECTIVE_YLIM_IGNORE_LABELS
    ]
    if not target_curves:
        target_curves = list(curves)

    common_length = _resolve_shortest_curve_length(target_curves)
    stacked = np.concatenate(
        [
            _maybe_smooth(series_config, values[:common_length]).reshape(-1)
            for series_config, values in target_curves
        ]
    )
    y_min = float(np.min(stacked))
    y_max = float(np.max(stacked))
    span = max(y_max - y_min, 1e-6)
    pad = max(span * OBJECTIVE_YLIM_PAD_RATIO, OBJECTIVE_YLIM_MIN_PAD)
    return y_min - pad, y_max + pad


def _apply_compare_header(fig, handles, labels, legend_ncol, layout_rect, legend_y):
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, float(legend_y)),
        ncol=int(legend_ncol),
        frameon=False,
        columnspacing=0.9,
        handlelength=1.8,
        handletextpad=0.45,
        labelspacing=0.25,
        borderaxespad=0.15,
    )
    fig.tight_layout(pad=0.28, rect=layout_rect)


def _save_figure(fig, stem):
    png_path = Path(build_compare_artifact_path(str(BASE_DIR), f"{stem}.png", seed=PREFERRED_SEED))
    pdf_path = Path(build_compare_artifact_path(str(BASE_DIR), f"{stem}.pdf", seed=PREFERRED_SEED))
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
    for series_config, values in curves:
        plot_values = _maybe_smooth(series_config, values[:common_length])
        ax.plot(
            episodes,
            plot_values,
            color=str(series_config["color"]),
            marker=str(series_config["marker"]),
            markevery=MARK_EVERY,
            label=_series_label(series_config),
        )
    return common_length


def _plot_objective(curves):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    max_episode = _plot_algorithm_curves(ax, curves)
    ax.set_xlabel("Episode")
    ax.set_ylabel(OBJECTIVE_LABEL)
    ax.set_xlim(1, max_episode)
    ax.set_ylim(*_resolve_objective_ylim(curves))
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _apply_compare_header(fig, handles, labels, COMPARE_LEGEND_NCOL, COMPARE_HEADER_RECT, COMPARE_LEGEND_Y)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_objective_ieee")


def _plot_cost(curves):
    fig, ax = plt.subplots(figsize=(FIG_WIDTH, FIG_HEIGHT))
    max_episode = _plot_algorithm_curves(ax, curves)
    ax.axhline(
        CONSTRAINT_LIMIT,
        color=REFERENCE_LINE_COLOR,
        linestyle="--",
        linewidth=REFERENCE_LINE_WIDTH,
        label="Constraint limit",
    )
    ax.set_xlabel("Episode")
    ax.set_ylabel(COST_LABEL)
    ax.set_xlim(1, max_episode)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _apply_compare_header(fig, handles, labels, COMPARE_LEGEND_NCOL, COMPARE_HEADER_RECT, COMPARE_LEGEND_Y)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_cost_ieee")


def _plot_reuse(reuse_plots):
    if not reuse_plots:
        raise ValueError("reuse_plots must contain at least one series.")

    fig_height = REUSE_PANEL_HEIGHT * max(len(reuse_plots), 1)
    fig, axes = plt.subplots(len(reuse_plots), 1, figsize=(FIG_WIDTH, fig_height), squeeze=False)
    axes = axes.reshape(-1)
    legend_handles = None
    legend_labels = None

    for idx, (series_label, episodes, new_policy, dk_policy, old_policy) in enumerate(reuse_plots):
        common_length, (episodes, new_policy, dk_policy, old_policy) = _truncate_plot_series(
            episodes,
            new_policy,
            dk_policy,
            old_policy,
        )
        ax = axes[idx]
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
        ax.set_title("{0} policy mixing weights".format(series_label), pad=4.0, fontsize=8.5, fontweight="semibold")
        _style_axis(ax)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, REUSE_LEGEND_Y),
            ncol=REUSE_LEGEND_NCOL,
            frameon=False,
            columnspacing=0.9,
            handlelength=1.8,
            handletextpad=0.45,
            labelspacing=0.25,
            borderaxespad=0.15,
        )
    fig.tight_layout(pad=0.28, rect=REUSE_HEADER_RECT)
    return _save_figure(fig, f"{OUTPUT_PREFIX}_reuse_ieee")


def main():
    _apply_ieee_style()
    reuse_series = _validate_plot_series_config(PLOT_SERIES)

    objective_curves = []
    cost_curves = []

    for series_config in PLOT_SERIES:
        reward_curve = _try_load_curve(
            lambda series_config=series_config: _resolve_series_mat_path(series_config, "reward_stem")
        )
        cost_curve = _try_load_curve(
            lambda series_config=series_config: _resolve_series_mat_path(series_config, "cost_stem")
        )
        if reward_curve is None or cost_curve is None:
            continue
        objective_curves.append((series_config, reward_curve))
        cost_curves.append((series_config, cost_curve))

    if not objective_curves or not cost_curves:
        raise FileNotFoundError("no CLQR curves were found in outputs or legacy root.")

    outputs = []
    outputs.extend(_plot_objective(objective_curves))
    outputs.extend(_plot_cost(cost_curves))

    reuse_plots = []
    for reuse_config in reuse_series:
        reuse_data = _try_load_reuse_history(
            lambda reuse_config=reuse_config: _resolve_series_mat_path(reuse_config, "rho_stem")
        )
        if reuse_data is None:
            continue
        episodes = np.arange(1, len(reuse_data[0]) + 1, dtype=np.int32)
        reuse_plots.append((_series_label(reuse_config), episodes, *reuse_data))
    if reuse_plots:
        outputs.extend(_plot_reuse(reuse_plots))

    for path in outputs:
        print(path)


if __name__ == "__main__":
    main()
