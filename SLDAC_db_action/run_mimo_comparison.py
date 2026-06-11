from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import subprocess
import sys
from argparse import Namespace
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat


# 对比仿真参数：默认对齐 SLDAC_code/MIMO1 的 MIMO 主入口，可直接改这里。
T = 500
NUM_NEW_DATA = 100
WINDOW = 10000
GRAD_T = T
EPISODE = 100
UPDATE_TIME_PER_EPISODE = 10
NUM_UPDATE_TIME = EPISODE * UPDATE_TIME_PER_EPISODE
Q_UPDATE_TIME = 1
MAX_STEPS = 2 * T + NUM_UPDATE_TIME * NUM_NEW_DATA

# 步长与 surrogate 参数：沿用 SLDAC_code/MIMO1 默认值，两个分支完全一致。
ALPHA_POW = 0.6
BETA_POW = 0.7
ETA_POW = 0.01
GAMMA_POW_REWARD = 0.3
GAMMA_POW_COST = 0.3
TAU_REWARD = 1
TAU_COST = 1

# 运行范围：可选 "both"（新旧都跑）、"new_only"（只跑 SNR-dB 新算法）、"old_only"（只跑旧 power 动作）。
RUN_SELECTION = "new_only"

# 打印间隔：每隔多少个记录 episode 打印一次结果；设为 0 可关闭逐点打印。
PRINT_INTERVAL = 1


THIS_DIR = Path(__file__).resolve().parent
ROOT_DIR = THIS_DIR.parent
DB_MIMO_DIR = THIS_DIR / "MIMO1"
LEGACY_MIMO_DIR = ROOT_DIR / "SLDAC_code" / "MIMO1"
OUTPUT_ROOT = THIS_DIR / "outputs"
ALL_MODES = ("legacy_power", "snr_db")


def selected_modes():
	selection = str(RUN_SELECTION).strip().lower()
	if selection in ("both", "all", "compare"):
		return ALL_MODES
	if selection in ("new", "new_only", "snr", "snr_db", "db_action"):
		return ("snr_db",)
	if selection in ("old", "old_only", "legacy", "legacy_power", "power"):
		return ("legacy_power",)
	raise ValueError("unsupported RUN_SELECTION: {0!r}".format(RUN_SELECTION))


def build_args():
	return Namespace(
		T=T,
		grad_T=GRAD_T,
		window=WINDOW,
		num_new_data=NUM_NEW_DATA,
		episode=EPISODE,
		update_time_per_episode=UPDATE_TIME_PER_EPISODE,
		num_update_time=NUM_UPDATE_TIME,
		Q_update_time=Q_UPDATE_TIME,
		MAX_STEPS=MAX_STEPS,
		alpha_pow=ALPHA_POW,
		beta_pow=BETA_POW,
		eta_pow=ETA_POW,
		gamma_pow_reward=GAMMA_POW_REWARD,
		gamma_pow_cost=GAMMA_POW_COST,
		tau_reward=TAU_REWARD,
		tau_cost=TAU_COST,
		print_interval=PRINT_INTERVAL,
	)


def config_dict():
	return {
		"T": T,
		"num_new_data": NUM_NEW_DATA,
		"window": WINDOW,
		"grad_T": GRAD_T,
		"episode": EPISODE,
		"update_time_per_episode": UPDATE_TIME_PER_EPISODE,
		"num_update_time": NUM_UPDATE_TIME,
		"Q_update_time": Q_UPDATE_TIME,
		"MAX_STEPS": MAX_STEPS,
		"alpha_pow": ALPHA_POW,
		"beta_pow": BETA_POW,
		"eta_pow": ETA_POW,
		"gamma_pow_reward": GAMMA_POW_REWARD,
		"gamma_pow_cost": GAMMA_POW_COST,
		"tau_reward": TAU_REWARD,
		"tau_cost": TAU_COST,
		"cssca_solver": "dual",
		"run_selection": RUN_SELECTION,
		"selected_modes": list(selected_modes()),
		"print_interval": PRINT_INTERVAL,
	}


def clear_mimo_modules():
	for name in ("SLDAC", "environment", "model", "critic_opt", "buffer", "utils"):
		sys.modules.pop(name, None)


def preload_dual_utils_for_legacy():
	spec = importlib.util.spec_from_file_location("utils", DB_MIMO_DIR / "utils.py")
	if spec is None or spec.loader is None:
		raise RuntimeError("failed to load dual utils module")
	module = importlib.util.module_from_spec(spec)
	sys.modules["utils"] = module
	spec.loader.exec_module(module)


def import_sldac_main(mode):
	clear_mimo_modules()
	if mode == "legacy_power":
		preload_dual_utils_for_legacy()
		sys.path.insert(0, str(LEGACY_MIMO_DIR))
	elif mode == "snr_db":
		sys.path.insert(0, str(DB_MIMO_DIR))
	else:
		raise ValueError("unsupported mode: {0}".format(mode))
	from SLDAC import SLDAC_main

	return SLDAC_main


def run_worker(mode, output_dir):
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)
	print("[{0}] start: output_dir={1}".format(mode, output_dir), flush=True)
	sldac_main = import_sldac_main(mode)
	objective_cost, avg_delay = sldac_main(build_args(), "MIMO")
	objective_cost = np.asarray(objective_cost, dtype=np.float64).reshape(-1)
	avg_delay = np.asarray(avg_delay, dtype=np.float64).reshape(-1)
	episodes = np.arange(1, objective_cost.size + 1)

	csv_path = output_dir / "{0}_metrics.csv".format(mode)
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["episode", "objective_cost_real_power", "avg_delay_per_user"])
		for episode, objective, delay in zip(episodes, objective_cost, avg_delay):
			writer.writerow([int(episode), float(objective), float(delay)])

	mat_path = output_dir / "{0}_metrics.mat".format(mode)
	savemat(
		mat_path,
		{
			"episode": episodes,
			"objective_cost_real_power": objective_cost,
			"avg_delay_per_user": avg_delay,
		},
	)

	summary = {
		"mode": mode,
		"config": config_dict(),
		"num_points": int(objective_cost.size),
		"final_objective_cost_real_power": float(objective_cost[-1]) if objective_cost.size else None,
		"final_avg_delay_per_user": float(avg_delay[-1]) if avg_delay.size else None,
		"csv": str(csv_path),
		"mat": str(mat_path),
	}
	summary_path = output_dir / "{0}_summary.json".format(mode)
	summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(summary, ensure_ascii=False), flush=True)


def print_interval_metrics(mode, episodes, objective_cost, avg_delay):
	interval = int(PRINT_INTERVAL)
	if interval <= 0:
		return
	total = int(len(episodes))
	for idx, episode in enumerate(episodes, start=1):
		if idx == 1 or idx % interval == 0 or idx == total:
			print(
				"[{0}] episode={1} objective_cost_real_power={2:.6g} avg_delay_per_user={3:.6g}".format(
					mode,
					int(episode),
					float(objective_cost[idx - 1]),
					float(avg_delay[idx - 1]),
				),
				flush=True,
			)


def load_mode_csv(path):
	rows = []
	with Path(path).open("r", newline="", encoding="utf-8") as f:
		for row in csv.DictReader(f):
			rows.append(
				{
					"episode": int(row["episode"]),
					"objective_cost_real_power": float(row["objective_cost_real_power"]),
					"avg_delay_per_user": float(row["avg_delay_per_user"]),
				}
			)
	return rows


def plot_comparison(run_dir, summaries, modes):
	colors = {"legacy_power": "#1f77b4", "snr_db": "#d62728"}
	fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
	for mode in modes:
		rows = load_mode_csv(summaries[mode]["csv"])
		x = [row["episode"] for row in rows]
		axes[0].plot(
			x,
			[row["objective_cost_real_power"] for row in rows],
			color=colors[mode],
			linewidth=1.8,
			marker="o",
			markersize=3,
			label=mode,
		)
		axes[1].plot(
			x,
			[row["avg_delay_per_user"] for row in rows],
			color=colors[mode],
			linewidth=1.8,
			marker="o",
			markersize=3,
			label=mode,
		)
	axes[0].set_title("Objective cost: real sum(power)")
	axes[0].set_xlabel("reported episode")
	axes[0].set_ylabel("real power")
	axes[0].grid(True, alpha=0.25)
	axes[1].set_title("Average delay per user")
	axes[1].set_xlabel("reported episode")
	axes[1].set_ylabel("delay")
	axes[1].grid(True, alpha=0.25)
	axes[0].legend(loc="best")
	if len(modes) > 1:
		fig.suptitle("SLDAC MIMO dual comparison: legacy power action vs SNR-dB action")
	else:
		fig.suptitle("SLDAC MIMO dual run: {0}".format(modes[0]))
	fig.tight_layout(rect=(0, 0, 1, 0.94))
	png_path = Path(run_dir) / "mimo_dual_action_comparison.png"
	pdf_path = Path(run_dir) / "mimo_dual_action_comparison.pdf"
	fig.savefig(png_path)
	fig.savefig(pdf_path)
	plt.close(fig)
	return png_path, pdf_path


def run_parent():
	run_dir = OUTPUT_ROOT / datetime.now().strftime("mimo_dual_action_compare_%Y%m%d_%H%M%S")
	run_dir.mkdir(parents=True, exist_ok=False)
	modes = selected_modes()
	print("selected_modes:", ", ".join(modes), flush=True)
	print("print_interval:", PRINT_INTERVAL, flush=True)
	summaries = {}
	for mode in modes:
		mode_dir = run_dir / mode
		command = [
			sys.executable,
			str(Path(__file__).resolve()),
			"--worker",
			mode,
			"--output-dir",
			str(mode_dir),
		]
		subprocess.run(command, cwd=str(ROOT_DIR), check=True)
		summary_path = mode_dir / "{0}_summary.json".format(mode)
		summaries[mode] = json.loads(summary_path.read_text(encoding="utf-8"))

	png_path, pdf_path = plot_comparison(run_dir, summaries, modes)
	summary = {
		"config": config_dict(),
		"modes": summaries,
		"plot_png": str(png_path),
		"plot_pdf": str(pdf_path),
	}
	(run_dir / "comparison_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
	print("output_dir:", run_dir)
	print("plot_png:", png_path)
	print("plot_pdf:", pdf_path)


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("--worker", choices=ALL_MODES)
	parser.add_argument("--output-dir")
	args = parser.parse_args()
	if args.worker:
		if not args.output_dir:
			raise ValueError("--output-dir is required in worker mode")
		run_worker(args.worker, args.output_dir)
	else:
		run_parent()


if __name__ == "__main__":
	main()
