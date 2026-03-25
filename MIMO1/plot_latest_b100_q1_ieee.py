import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import loadmat

from artifact_paths import build_compare_artifact_path, resolve_algorithm_artifact_path


BASE_DIR = Path(__file__).resolve().parent

# 绘图结果配置：脚本固定服务于 b100_q1，对外只暴露顶部配置区。
RUN_TAG = "b100_q1"
PREFERRED_SEED = 0
OUTPUT_PREFIX = f"MIMO_seed{PREFERRED_SEED}_algorithms_compare_{RUN_TAG}"

# 物理量配置：objective 对应总发射功率，cost 对应平均用户时延积压上限。
DELAY_LIMIT = 1.2
OBJECTIVE_LABEL = "Objective cost (avg. total transmit power)"
COST_LABEL = "Constraint cost (avg. user delay backlog)"
REUSE_LABEL = "Policy reuse probability"

# MIMO 特有展示配置：仅允许 HRL objective 曲线使用可视化偏移，图例中不显示偏移文本。
HRL_OBJECTIVE_OFFSET = 0.3

# 绘图目标配置：直接在这里增删曲线即可，不依赖 CLI。
PLOT_SERIES = [
    {
        "label": "Fused-CPRO",
        "artifact_group": "Fused_CPRO",
        "reward_stem": f"Fused_CPRO_reward_{RUN_TAG}.mat",
        "cost_stem": f"Fused_CPRO_cost_{RUN_TAG}.mat",
        "rho_stem": f"Fused_CPRO_rho_{RUN_TAG}.mat",
        "color": "#0072B2",
        "marker": "o",
        "prefer_seed_suffix": True,
    },
    {
        "label": "HRL",
        "artifact_group": "HRL",
        "reward_stem": f"HRL_reward_{RUN_TAG}.mat",
        "cost_stem": f"HRL_cost_{RUN_TAG}.mat",
        "rho_stem": f"HRL_rho_{RUN_TAG}.mat",
        "color": "#E69F00",
        "marker": "P",
        "prefer_seed_suffix": True,
        "objective_offset": HRL_OBJECTIVE_OFFSET,
    },
    {
        "label": "SLDAC",
        "artifact_group": "SLDAC",
        "reward_stem": f"SLDAC_reward_{RUN_TAG}.mat",
        "cost_stem": f"SLDAC_cost_{RUN_TAG}.mat",
        "color": "#D55E00",
        "marker": "s",
        "prefer_seed_suffix": True,
    },
    {
        "label": "SCAOPO",
        "artifact_group": "SCAOPO",
        "reward_stem": "SCAOPO_reward_100.mat",
        "cost_stem": "SCAOPO_cost_100.mat",
        "color": "#009E73",
        "marker": "^",
        "prefer_seed_suffix": False,
    },
    {
        "label": "PPO",
        "artifact_group": "ppo",
        "reward_stem": "reward_ppo_100.mat",
        "cost_stem": "cost_ppo_100.mat",
        "color": "#CC79A7",
        "marker": "D",
        "prefer_seed_suffix": False,
    },
    {
        "label": "CPO",
        "artifact_group": "cpo",
        "reward_stem": "reward_cpo_100.mat",
        "cost_stem": "cost_cpo_100.mat",
        "color": "#7E2F8E",
        "marker": "v",
        "prefer_seed_suffix": False,
    },
]

# 科研绘图样式：单栏尺寸、细网格、较高分辨率，适合论文直接引用。
FIG_WIDTH = 4.6
FIG_HEIGHT = 3.35
FIG_DPI = 600
LINE_WIDTH = 1.45
REFERENCE_LINE_WIDTH = 1.15
MARKER_SIZE = 5.0
MARK_EVERY = 6
COMPARE_LEGEND_NCOL = 4
REUSE_LEGEND_NCOL = 3
COMPARE_HEADER_RECT = (0.0, 0.0, 1.0, 0.78)
REUSE_HEADER_RECT = (0.0, 0.0, 1.0, 0.88)
REUSE_PANEL_HEIGHT = 2.6

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
SERIES_OPTIONAL_KEYS = ("rho_stem", "objective_offset")
SEED_SUFFIX_PATTERN = re.compile(r"_seed\d+$")


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


def _series_label(series_config):
    return str(series_config["label"]).strip()


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


def _build_seed_preferred_stem(stem):
    stem_path = Path(str(stem))
    if SEED_SUFFIX_PATTERN.search(stem_path.stem):
        return str(stem_path)
    return "{0}_seed{1}{2}".format(stem_path.stem, PREFERRED_SEED, stem_path.suffix)


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
            if str(series_config["artifact_group"]).strip() not in ("Fused_CPRO", "HRL"):
                raise ValueError("Only Fused_CPRO or HRL series may define rho_stem. Invalid label: {0}".format(label))
            reuse_series.append(series_config)

        objective_offset = series_config.get("objective_offset")
        if objective_offset is not None:
            if label != "HRL":
                raise ValueError("Only HRL series may define objective_offset. Invalid label: {0}".format(label))
            try:
                float(objective_offset)
            except (TypeError, ValueError) as exc:
                raise TypeError(
                    "PLOT_SERIES[{0}] field objective_offset must be numeric.".format(idx)
                ) from exc

    return reuse_series


def _resolve_series_mat_path(series_config, stem_key):
    artifact_group = str(series_config["artifact_group"])
    stem = str(series_config[stem_key])
    candidates = []
    if bool(series_config["prefer_seed_suffix"]):
        candidates.append(_build_seed_preferred_stem(stem))
    candidates.append(stem)

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
    ax.grid(True, axis="x", alpha=0.12)
    for spine in ax.spines.values():
        spine.set_linewidth(0.9)


def _apply_compare_header(fig, handles, labels, title_text, legend_ncol, layout_rect):
    fig.suptitle(title_text, y=0.985)
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.93),
        ncol=int(legend_ncol),
        frameon=False,
        columnspacing=1.1,
        handlelength=2.0,
    )
    fig.tight_layout(pad=0.45, rect=layout_rect)


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


def _objective_plot_values(series_config, values):
    values = np.asarray(values, dtype=np.float64)
    offset = float(series_config.get("objective_offset", 0.0))
    if offset == 0.0:
        return values
    return values + offset


def _plot_algorithm_curves(ax, curves, transform_fn=None):
    common_length = _resolve_shortest_curve_length(curves)
    episodes = np.arange(1, common_length + 1, dtype=np.int32)
    for series_config, values in curves:
        plot_values = values[:common_length]
        if transform_fn is not None:
            plot_values = transform_fn(series_config, plot_values)
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
    max_episode = _plot_algorithm_curves(ax, curves, transform_fn=_objective_plot_values)
    ax.set_xlabel("Episode")
    ax.set_ylabel(OBJECTIVE_LABEL)
    ax.set_xlim(1, max_episode)
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _apply_compare_header(fig, handles, labels, "MIMO, b = 100, q = 1", COMPARE_LEGEND_NCOL, COMPARE_HEADER_RECT)
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
    _style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    _apply_compare_header(fig, handles, labels, "MIMO, b = 100, q = 1", COMPARE_LEGEND_NCOL, COMPARE_HEADER_RECT)
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
        ax.set_title("{0} policy mixing weights".format(series_label))
        _style_axis(ax)
        if legend_handles is None:
            legend_handles, legend_labels = ax.get_legend_handles_labels()

    if legend_handles is not None and legend_labels is not None:
        fig.legend(
            legend_handles,
            legend_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.985),
            ncol=REUSE_LEGEND_NCOL,
            frameon=False,
            columnspacing=1.2,
            handlelength=2.0,
        )
    fig.tight_layout(pad=0.45, rect=REUSE_HEADER_RECT)
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
        raise FileNotFoundError("no MIMO curves were found in outputs or legacy root.")

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
