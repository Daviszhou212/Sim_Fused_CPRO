from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.io import savemat

from .plot_results import plot_curves, plot_timing
from .span_sldac_runner import SpanSldacConfig, run_span_sldac


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _log(verbose: bool, message: str):
    if verbose:
        print(message, flush=True)


def run_experiment(
    output_root: Path,
    run_name: str,
    solvers=("full", "span", "active"),
    seed=0,
    T=500,
    grad_T=None,
    num_new_data=100,
    episode=60,
    update_time_per_episode=10,
    num_update_time=None,
    q_update_time=1,
    window=10000,
    print_interval=1,
    make_plots=True,
    verbose=True,
):
    output_root = Path(output_root)
    run_dir = output_root / str(run_name)
    run_dir.mkdir(parents=True, exist_ok=True)
    resolved_grad_T = int(T if grad_T is None else grad_T)
    resolved_num_update_time = int(
        episode * update_time_per_episode if num_update_time is None else num_update_time
    )
    config = SpanSldacConfig(
        seed=int(seed),
        T=int(T),
        grad_T=resolved_grad_T,
        num_new_data=int(num_new_data),
        episode=int(episode),
        update_time_per_episode=int(update_time_per_episode),
        num_update_time=resolved_num_update_time,
        q_update_time=int(q_update_time),
        window=int(window),
        print_interval=max(1, int(print_interval)),
    )
    _log(
        verbose,
        (
            f"[Spantest] output_dir={run_dir} solvers={' '.join(solvers)} "
            f"T={config.T} grad_T={config.grad_T} num_new_data={config.num_new_data} "
            f"episodes={config.episode} update_time_per_episode={config.update_time_per_episode} "
            f"num_update_time={config.num_update_time} print_interval={config.print_interval}"
        ),
    )

    results = []
    for solver in solvers:
        _log(verbose, f"[Spantest] solver={solver} start")
        item = run_span_sldac(solver, config, verbose=verbose)
        results.append(item)
        solve_times = [float(row["solve_time_sec"]) for row in item["timing_rows"]]
        mean_solve = "" if not solve_times else f"{float(np.mean(solve_times)):.6f}"
        _log(
            verbose,
            (
                f"[Spantest] solver={solver} finished "
                f"actor_updates={len(item['timing_rows'])} mean_solve_time_sec={mean_solve}"
            ),
        )

    timing_rows = []
    summary_rows = []
    mat_payload = {}
    for item in results:
        solver = item["solver"]
        reward = item["reward_curve"]
        cost = item["cost_curve"]
        mat_payload[f"{solver}_objective_cost"] = reward
        mat_payload[f"{solver}_constraint_cost"] = cost
        timing_rows.extend(item["timing_rows"])
        solve_times = [float(row["solve_time_sec"]) for row in item["timing_rows"]]
        dims = [int(row["decision_dim"]) for row in item["timing_rows"]]
        summary_rows.append(
            {
                "solver": solver,
                "theta_dim": item["theta_dim"],
                "num_logged_points": len(reward),
                "num_actor_updates": len(item["timing_rows"]),
                "final_objective_cost": "" if len(reward) == 0 else float(reward[-1]),
                "final_constraint_cost": "" if len(cost) == 0 else float(cost[-1]),
                "mean_solve_time_sec": "" if not solve_times else float(np.mean(solve_times)),
                "max_solve_time_sec": "" if not solve_times else float(np.max(solve_times)),
                "mean_decision_dim": "" if not dims else float(np.mean(dims)),
                "max_decision_dim": "" if not dims else int(np.max(dims)),
            }
        )

    _write_csv(run_dir / "timing.csv", timing_rows)
    _write_csv(run_dir / "summary.csv", summary_rows)
    savemat(run_dir / "curves.mat", mat_payload)
    metadata = {
        "config": config.__dict__,
        "solvers": list(solvers),
        "note": "reward_curve stores MIMO1 objective cost; lower is better.",
    }
    (run_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if make_plots:
        _log(verbose, f"[Spantest] plotting output_dir={run_dir}")
        plot_curves(results, run_dir)
        plot_timing(timing_rows, run_dir)
    _log(verbose, f"[Spantest] done output_dir={run_dir}")
    return run_dir


def main():
    parser = argparse.ArgumentParser(description="Run self-contained MIMO1 span-CSSCA comparison.")
    parser.add_argument("--output-root", type=Path, default=Path("Spantest") / "outputs")
    parser.add_argument("--run-name", type=str, default="mimo1_span_compare")
    parser.add_argument(
        "--solvers",
        nargs="+",
        default=["full", "span", "active"],
        choices=["full", "span", "active", "dual"],
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--T", type=int, default=500)
    parser.add_argument("--grad-T", dest="grad_T", type=int, default=None)
    parser.add_argument("--num-new-data", type=int, default=100)
    parser.add_argument("--episode", type=int, default=60)
    parser.add_argument("--update-time-per-episode", type=int, default=10)
    parser.add_argument("--num-update-time", type=int, default=None)
    parser.add_argument("--q-update-time", type=int, default=1)
    parser.add_argument("--window", type=int, default=10000)
    parser.add_argument("--print-interval", type=int, default=1)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    run_dir = run_experiment(
        output_root=args.output_root,
        run_name=args.run_name,
        solvers=tuple(args.solvers),
        seed=args.seed,
        T=args.T,
        grad_T=args.grad_T,
        num_new_data=args.num_new_data,
        episode=args.episode,
        update_time_per_episode=args.update_time_per_episode,
        num_update_time=args.num_update_time,
        q_update_time=args.q_update_time,
        window=args.window,
        print_interval=args.print_interval,
        make_plots=not args.no_plots,
    )
    print(run_dir)


if __name__ == "__main__":
    main()
