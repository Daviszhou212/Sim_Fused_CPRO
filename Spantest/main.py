from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

if __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# 默认输出目录：所有 mat/png/pdf/csv/json 会写入该目录下的 run_name 子目录。
DEFAULT_OUTPUT_ROOT = Path("Spantest") / "outputs"

# 默认实验名：正式仿真时建议改成能表达日期、场景和参数规模的名字。
DEFAULT_RUN_NAME = "20260610"

# 是否运行原版 SLDAC full-parameter CSSCA；关闭可显著节省仿真时间。
RUN_ORIGINAL_SLDAC = False

# 是否运行 full-gradient span 低维 CSSCA；这是当前主要对比方法。
RUN_LOWER_DIMENSION_SLDAC = False

# 是否运行 active-gradient span 低维 CSSCA；用于观察进一步降维后的安全更新效果。
RUN_ACTIVE_SLDAC = False

# 是否运行结构化拉格朗日 SLDAC；用于替代高维 CVXPY 求解原版 CSSCA。
RUN_LAGRANGIAN_SLDAC = True

# 默认求解器集合：由上方开关派生，直接运行 main.py 时生效。
DEFAULT_SOLVERS = tuple(
    solver
    for enabled, solver in (
        (RUN_ORIGINAL_SLDAC, "full"),
        (RUN_LOWER_DIMENSION_SLDAC, "span"),
        (RUN_ACTIVE_SLDAC, "active"),
        (RUN_LAGRANGIAN_SLDAC, "dual"),
    )
    if enabled
)

# 是否默认生成 png/pdf 图：关闭后仍会写 csv/mat/json。
DEFAULT_MAKE_PLOTS = True

# 与 SLDAC_code/MIMO1/MIMO_main.py 保持一致的主实验参数。
SEED = 0
T = 500
NUM_NEW_DATA = 100
WINDOW = 10000
GRAD_T = T
EPISODE = 60
UPDATE_TIME_PER_EPISODE = 10
NUM_UPDATE_TIME = EPISODE * UPDATE_TIME_PER_EPISODE
Q_UPDATE_TIME = 1

# 进度打印间隔：每隔多少个 episode 打印一次；1 表示每个 episode 都打印。
PRINT_INTERVAL = 5

SLDAC_MIMO1_CONFIG = {
    "seed": SEED,
    "T": T,
    "grad_T": GRAD_T,
    "num_new_data": NUM_NEW_DATA,
    "episode": EPISODE,
    "update_time_per_episode": UPDATE_TIME_PER_EPISODE,
    "num_update_time": NUM_UPDATE_TIME,
    "q_update_time": Q_UPDATE_TIME,
    "window": WINDOW,
    "print_interval": PRINT_INTERVAL,
}

# 小规模链路检查配置：只用于测试入口、绘图和文件写出，不作为正式对比参数。
SMOKE_CONFIG = {
    "seed": SEED,
    "T": 5,
    "grad_T": 5,
    "num_new_data": 5,
    "episode": 2,
    "update_time_per_episode": 1,
    "num_update_time": 2,
    "q_update_time": 1,
    "window": 20,
    "print_interval": 1,
}


def _normalize_solvers(solvers: Iterable[str] | None) -> tuple[str, ...]:
    if solvers is None:
        normalized = tuple(DEFAULT_SOLVERS)
    else:
        normalized = tuple(str(item).strip() for item in solvers if str(item).strip())
    if not normalized:
        raise ValueError("solvers must not be empty; enable at least one RUN_* switch in Spantest/main.py")
    unsupported = sorted(set(normalized) - {"full", "span", "active", "dual"})
    if unsupported:
        raise ValueError(f"unsupported solvers: {unsupported}")
    return normalized


def run_spantest_main(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
    solvers: Iterable[str] | None = None,
    make_plots: bool = DEFAULT_MAKE_PLOTS,
    use_smoke_config: bool = False,
    verbose: bool = True,
):
    config = dict(SMOKE_CONFIG if use_smoke_config else SLDAC_MIMO1_CONFIG)
    if verbose:
        print("[Spantest] loading dependencies", flush=True)
    from Spantest.run_experiment import run_experiment

    run_dir = run_experiment(
        output_root=Path(output_root),
        run_name=str(run_name),
        solvers=_normalize_solvers(solvers),
        make_plots=bool(make_plots),
        verbose=bool(verbose),
        **config,
    )
    return run_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="All-in-one entry for self-contained MIMO1 span-CSSCA comparison."
    )
    parser.add_argument("--smoke", action="store_true", help="Use the small smoke-test config.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=list(DEFAULT_SOLVERS),
        choices=["full", "span", "active", "dual"],
        help="Keep all three for full comparison, or pass a subset for quick checks.",
    )
    parser.add_argument("--no-plots", action="store_true", help="Write csv/mat/json only.")
    return parser


def main(argv: list[str] | None = None) -> Path:
    args = build_parser().parse_args(argv)
    run_dir = run_spantest_main(
        output_root=args.output_root,
        run_name=args.run_name,
        solvers=tuple(args.solvers),
        make_plots=not args.no_plots,
        use_smoke_config=args.smoke,
    )
    print(run_dir)
    return run_dir


if __name__ == "__main__":
    main()
