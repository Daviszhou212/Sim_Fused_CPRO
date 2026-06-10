from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .cssca_solvers import CsscaProblem, solve_gradient_span_cvxpy
from .dual_solvers import solve_cssca_dual


# 输出目录：benchmark 不跑环境训练，只保存合成 CSSCA 子问题的时间结果。
DEFAULT_OUTPUT_ROOT = Path("Spantest") / "outputs"

# 默认输出子目录名；会写 summary.csv 和 png/pdf 图。
DEFAULT_RUN_NAME = "large_actor_dual_benchmark"

# 需要测试的 actor 参数维度；可在文件顶部直接改。
THETA_DIMS = (25994, 100000, 300000, 1000000)

# MIMO1 对应 4 个约束；总梯度行数为 1 个 objective + 4 个 constraints。
CONSTRAINT_DIM = 4

# 每个维度重复次数；越大越稳，但大维度会更耗时。
REPEATS = 3

# 只在较小维度上额外跑 span CVXPY，避免高维 SVD/CVXPY 检查浪费时间。
SPAN_CVXPY_MAX_DIM = 100000


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def make_synthetic_problem(theta_dim: int, constraint_dim: int, seed: int) -> tuple[CsscaProblem, float]:
    rng = np.random.default_rng(seed)
    t0 = time.perf_counter()
    gradients = rng.normal(size=(constraint_dim + 1, theta_dim)).astype(np.float64)
    gradients /= np.linalg.norm(gradients, axis=1, keepdims=True) + 1e-12
    values = np.empty(constraint_dim + 1, dtype=np.float64)
    values[0] = 0.25
    values[1:] = np.linspace(-0.02, 0.04, constraint_dim)
    tau = np.ones(constraint_dim + 1, dtype=np.float64)
    theta = np.zeros(theta_dim, dtype=np.float64)
    generation_time = time.perf_counter() - t0
    return CsscaProblem(values=values, gradients=gradients, theta=theta, tau=tau), generation_time


def run_benchmark(
    output_root: Path = DEFAULT_OUTPUT_ROOT,
    run_name: str = DEFAULT_RUN_NAME,
    theta_dims=THETA_DIMS,
    constraint_dim: int = CONSTRAINT_DIM,
    repeats: int = REPEATS,
    span_cvxpy_max_dim: int = SPAN_CVXPY_MAX_DIM,
):
    run_dir = Path(output_root) / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for theta_dim in tuple(int(item) for item in theta_dims):
        for repeat in range(int(repeats)):
            problem, generation_time = make_synthetic_problem(theta_dim, constraint_dim, seed=2026 + repeat)

            t0 = time.perf_counter()
            dual_result = solve_cssca_dual(problem)
            dual_wall_time = time.perf_counter() - t0

            span_time = ""
            step_error = ""
            objective_error = ""
            if theta_dim <= int(span_cvxpy_max_dim):
                t1 = time.perf_counter()
                span_result = solve_gradient_span_cvxpy(problem)
                span_time = time.perf_counter() - t1
                step_error = float(np.linalg.norm(span_result.step - dual_result.step))
                objective_error = abs(float(span_result.objective_value) - float(dual_result.objective_value))

            rows.append(
                {
                    "theta_dim": theta_dim,
                    "constraint_dim": int(constraint_dim),
                    "repeat": repeat,
                    "branch": dual_result.branch,
                    "status": dual_result.status,
                    "gradient_generation_sec": float(generation_time),
                    "dual_total_sec": float(dual_wall_time),
                    "dual_reported_solve_sec": float(dual_result.solve_time_sec),
                    "dual_decision_dim": int(dual_result.decision_dim),
                    "gradient_rank": int(dual_result.gradient_rank),
                    "span_cvxpy_sec": span_time,
                    "span_dual_step_error": step_error,
                    "span_dual_objective_error": objective_error,
                }
            )

    _write_csv(run_dir / "summary.csv", rows)
    plot_benchmark(rows, run_dir)
    return run_dir


def plot_benchmark(rows: list[dict], run_dir: Path):
    by_dim: dict[int, list[dict]] = {}
    for row in rows:
        by_dim.setdefault(int(row["theta_dim"]), []).append(row)
    dims = np.array(sorted(by_dim), dtype=np.int64)
    dual_mean = np.array([np.mean([float(item["dual_total_sec"]) for item in by_dim[int(dim)]]) for dim in dims])
    generation_mean = np.array(
        [np.mean([float(item["gradient_generation_sec"]) for item in by_dim[int(dim)]]) for dim in dims]
    )

    plt.figure(figsize=(8, 5))
    plt.plot(dims, dual_mean, marker="o", linewidth=2.0, label="dual solver total")
    plt.plot(dims, generation_mean, marker="s", linewidth=2.0, label="gradient generation")
    plt.xscale("log")
    plt.xlabel("actor parameter dimension")
    plt.ylabel("time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "dual_time_vs_actor_dim.png", dpi=200)
    plt.savefig(run_dir / "dual_time_vs_actor_dim.pdf")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Benchmark structured dual CSSCA solver on large actor dimensions.")
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", type=str, default=DEFAULT_RUN_NAME)
    parser.add_argument("--theta-dims", nargs="+", type=int, default=list(THETA_DIMS))
    parser.add_argument("--repeats", type=int, default=REPEATS)
    parser.add_argument("--span-cvxpy-max-dim", type=int, default=SPAN_CVXPY_MAX_DIM)
    args = parser.parse_args()
    run_dir = run_benchmark(
        output_root=args.output_root,
        run_name=args.run_name,
        theta_dims=tuple(args.theta_dims),
        repeats=args.repeats,
        span_cvxpy_max_dim=args.span_cvxpy_max_dim,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
