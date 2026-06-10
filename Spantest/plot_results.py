from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_curves(results, run_dir: Path):
    run_dir = Path(run_dir)
    x_max = max((len(item["reward_curve"]) for item in results), default=0)
    x = np.arange(x_max)
    plt.figure(figsize=(8, 5))
    for item in results:
        y = item["reward_curve"]
        plt.plot(x[: len(y)], y, marker="o", linewidth=1.8, label=item["solver"])
    plt.xlabel("logged episode")
    plt.ylabel("objective cost (lower is better)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "objective_cost.png", dpi=160)
    plt.savefig(run_dir / "objective_cost.pdf")
    plt.close()

    plt.figure(figsize=(8, 5))
    for item in results:
        y = item["cost_curve"]
        plt.plot(x[: len(y)], y, marker="o", linewidth=1.8, label=item["solver"])
    plt.axhline(1.2, color="black", linestyle="--", linewidth=1.2, label="average cost limit")
    plt.xlabel("logged episode")
    plt.ylabel("average delay per user")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "constraint_cost.png", dpi=160)
    plt.savefig(run_dir / "constraint_cost.pdf")
    plt.close()


def plot_timing(timing_rows, run_dir: Path):
    run_dir = Path(run_dir)
    by_solver = {}
    for row in timing_rows:
        by_solver.setdefault(row["solver"], []).append(row)

    plt.figure(figsize=(8, 5))
    for solver, rows in by_solver.items():
        updates = [int(row["update"]) for row in rows]
        values = [float(row["solve_time_sec"]) for row in rows]
        plt.plot(updates, values, marker="o", linewidth=1.8, label=solver)
    plt.xlabel("actor update")
    plt.ylabel("CSSCA solve time (s)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "solve_time.png", dpi=160)
    plt.savefig(run_dir / "solve_time.pdf")
    plt.close()

    plt.figure(figsize=(8, 5))
    for solver, rows in by_solver.items():
        updates = [int(row["update"]) for row in rows]
        values = [int(row["decision_dim"]) for row in rows]
        plt.plot(updates, values, marker="o", linewidth=1.8, label=solver)
    plt.xlabel("actor update")
    plt.ylabel("CVXPY decision dimension")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(run_dir / "decision_dim.png", dpi=160)
    plt.savefig(run_dir / "decision_dim.pdf")
    plt.close()

