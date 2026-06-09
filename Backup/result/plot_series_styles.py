"""共享 MIMO / CLQR 绘图脚本的算法样式表。"""

STYLE_BY_LABEL = {
    "Fused-CPRO": {"color": "#0072B2", "marker": "o"},
    "Fused-CPRO-RhoNew": {"color": "#56B4E9", "marker": "d"},
    "Fused-CPRO-CosRho": {"color": "#2A9D8F", "marker": "h"},
    "HRL": {"color": "#E69F00", "marker": "P"},
    "PRCRL": {"color": "#D55E00", "marker": "X"},
    "SLDAC": {"color": "#009E73", "marker": "s"},
    "DK": {"color": "#8C8C00", "marker": "*"},
    "ACPO": {"color": "#CC79A7", "marker": "D"},
    "SCAOPO": {"color": "#64B5CD", "marker": "^"},
    "PPO-Lag": {"color": "#000000", "marker": "<"},
    "CPO": {"color": "#7E2F8E", "marker": ">"},
}


def apply_series_style(series_config):
    if not isinstance(series_config, dict):
        raise TypeError("series_config must be a dict.")

    styled_config = dict(series_config)
    label = str(styled_config.get("label", "")).strip()
    if not label:
        raise ValueError("series_config label must be non-empty.")

    style = STYLE_BY_LABEL.get(label)
    if style is None:
        raise ValueError("Unknown plot series label for shared style table: {0}".format(label))

    styled_config["label"] = label
    styled_config["color"] = style["color"]
    styled_config["marker"] = style["marker"]
    return styled_config
