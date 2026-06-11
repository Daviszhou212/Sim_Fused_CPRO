from __future__ import annotations

import csv
import json
import sys
from argparse import Namespace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from scipy.io import savemat
from scipy.optimize import minimize


# 本脚本只用于本次 seed0 / 100 episode 仿真，不回写 SLDAC_code 源码。
# 参数沿用 SLDAC_code/MIMO1/MIMO_main.py 的默认 MIMO SLDAC 设置。
SEED = 0
T = 500
NUM_NEW_DATA = 100
WINDOW = 10000
GRAD_T = T
EPISODE = 100
UPDATE_TIME_PER_EPISODE = 10
NUM_UPDATE_TIME = EPISODE * UPDATE_TIME_PER_EPISODE
Q_UPDATE_TIME = 1
MAX_STEPS = 2 * T + NUM_UPDATE_TIME * NUM_NEW_DATA
ALPHA_POW = 0.6
BETA_POW = 0.7
ETA_POW = 0.01
GAMMA_POW_REWARD = 0.3
GAMMA_POW_COST = 0.3
TAU_REWARD = 1.0
TAU_COST = 1.0

# dual 子问题数值保护；不改变 SLDAC 训练主循环。
DUAL_EPS = 1e-12
DUAL_MAX_ITER = 500

OUT_DIR = Path(__file__).resolve().parent
MIMO_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(MIMO_DIR))


def _as_problem_arrays(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost):
	values = np.asarray(func_value_np, dtype=np.float64).reshape(-1)
	gradients = np.asarray(grad_np, dtype=np.float64)
	theta = np.asarray(paras_t_np, dtype=np.float64).reshape(-1)
	if gradients.ndim != 2:
		raise ValueError("grad_np must be a 2-D array")
	if values.size != gradients.shape[0]:
		raise ValueError("func_value_np and grad_np row mismatch")
	if theta.size != gradients.shape[1]:
		raise ValueError("paras_t_np and grad_np dimension mismatch")
	if values.size <= 0:
		raise ValueError("CSSCA problem must contain at least one objective row")
	tau_reward = float(tau_reward)
	tau_cost = float(tau_cost)
	if tau_reward <= 0.0 or tau_cost <= 0.0:
		raise ValueError("tau_reward and tau_cost must be positive")
	tau = tau_cost * np.ones(values.size, dtype=np.float64)
	tau[0] = tau_reward
	if (not np.isfinite(values).all()) or (not np.isfinite(gradients).all()) or (not np.isfinite(theta).all()):
		raise ValueError("CSSCA values, gradients, and parameters must be finite")
	return values, gradients, theta, tau


def _surrogate_values(values, gradients, tau, step):
	return values + gradients @ step + tau * float(step @ step)


def _solve_feasible_dual(values, gradients, tau):
	m = int(values.shape[0])
	if m <= 0:
		return np.zeros(gradients.shape[1], dtype=np.float64), np.zeros(0, dtype=np.float64), True
	if m == 1:
		lam = np.ones(1, dtype=np.float64)
		step = -gradients[0] / (2.0 * max(float(tau[0]), DUAL_EPS))
		return step, lam, True

	gram = gradients @ gradients.T

	def objective(lam):
		tau_lam = max(float(tau @ lam), DUAL_EPS)
		quad = float(lam @ gram @ lam)
		return -float(values @ lam - quad / (4.0 * tau_lam))

	def jacobian(lam):
		tau_lam = max(float(tau @ lam), DUAL_EPS)
		gram_lam = gram @ lam
		quad = float(lam @ gram_lam)
		grad = values - ((2.0 * gram_lam * tau_lam - quad * tau) / (4.0 * tau_lam * tau_lam))
		return -grad

	result = minimize(
		objective,
		np.full(m, 1.0 / float(m), dtype=np.float64),
		jac=jacobian,
		method="SLSQP",
		bounds=[(0.0, None)] * m,
		constraints=({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0), "jac": lambda x: np.ones_like(x)}),
		options={"ftol": 1e-10, "maxiter": DUAL_MAX_ITER, "disp": False},
	)
	lam = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
	lam_sum = float(np.sum(lam))
	if lam_sum <= DUAL_EPS:
		lam = np.full(m, 1.0 / float(m), dtype=np.float64)
	else:
		lam = lam / lam_sum
	tau_lam = max(float(tau @ lam), DUAL_EPS)
	step = -(lam @ gradients) / (2.0 * tau_lam)
	return step, lam, bool(result.success)


def _solve_objective_dual(values, gradients, tau):
	g0 = gradients[0]
	tau0 = max(float(tau[0]), DUAL_EPS)
	m = int(values.size - 1)
	if m <= 0:
		return -g0 / (2.0 * tau0), np.zeros(0, dtype=np.float64), True

	constraint_values = values[1:]
	constraint_gradients = gradients[1:]
	constraint_tau = tau[1:]
	gram_cc = constraint_gradients @ constraint_gradients.T
	gram_0c = constraint_gradients @ g0
	gram_00 = float(g0 @ g0)
	value0 = float(values[0])

	def objective(lam):
		tau_lam = max(tau0 + float(constraint_tau @ lam), DUAL_EPS)
		combo_quad = gram_00 + 2.0 * float(gram_0c @ lam) + float(lam @ gram_cc @ lam)
		return -float(value0 + constraint_values @ lam - combo_quad / (4.0 * tau_lam))

	def jacobian(lam):
		tau_lam = max(tau0 + float(constraint_tau @ lam), DUAL_EPS)
		gram_lam = gram_0c + gram_cc @ lam
		combo_quad = gram_00 + 2.0 * float(gram_0c @ lam) + float(lam @ gram_cc @ lam)
		grad = constraint_values - (
			(2.0 * gram_lam * tau_lam - combo_quad * constraint_tau) / (4.0 * tau_lam * tau_lam)
		)
		return -grad

	result = minimize(
		objective,
		np.zeros(m, dtype=np.float64),
		jac=jacobian,
		method="L-BFGS-B",
		bounds=[(0.0, None)] * m,
		options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": DUAL_MAX_ITER, "maxls": 50},
	)
	lam = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
	tau_lam = max(tau0 + float(constraint_tau @ lam), DUAL_EPS)
	step = -(g0 + lam @ constraint_gradients) / (2.0 * tau_lam)
	return step, lam, bool(result.success)


def solve_dual_cssca_update(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost, feasibility_tol=1e-8):
	values, gradients, theta, tau = _as_problem_arrays(
		func_value_np,
		grad_np,
		paras_t_np,
		tau_reward,
		tau_cost,
	)
	feasible_step, _, feasible_success = _solve_feasible_dual(values[1:], gradients[1:], tau[1:])
	feasible_surrogates = _surrogate_values(values, gradients, tau, feasible_step)
	feasible_x = -np.inf if values.size == 1 else float(np.max(feasible_surrogates[1:]))
	if feasible_x <= float(feasibility_tol):
		step, _, success = _solve_objective_dual(values, gradients, tau)
		branch = "objective"
	else:
		step, success = feasible_step, feasible_success
		branch = "feasible"
	if not np.isfinite(step).all():
		return theta.copy(), {"branch": "skip_nonfinite_step", "status": "skipped", "feasible_x": feasible_x}
	return theta + step, {
		"branch": branch,
		"status": "optimal" if success else "optimal_inaccurate",
		"feasible_x": feasible_x,
	}


def dual_update_policy(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost):
	paras_bar, _ = solve_dual_cssca_update(
		func_value_np,
		grad_np,
		paras_t_np,
		tau_reward=tau_reward,
		tau_cost=tau_cost,
	)
	return paras_bar


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
	)


def write_outputs(objective_cost, avg_delay):
	objective_cost = np.asarray(objective_cost, dtype=np.float64).reshape(1, -1)
	avg_delay = np.asarray(avg_delay, dtype=np.float64).reshape(1, -1)
	episodes = np.arange(1, objective_cost.size + 1)

	reward_mat = OUT_DIR / "SLDAC_dual_seed0_ep100_reward_b100_q1.mat"
	cost_mat = OUT_DIR / "SLDAC_dual_seed0_ep100_cost_b100_q1.mat"
	savemat(reward_mat, {"array": objective_cost, "objective_cost_real_power": objective_cost})
	savemat(cost_mat, {"array": avg_delay, "avg_delay_per_user": avg_delay})

	csv_path = OUT_DIR / "SLDAC_dual_seed0_ep100_metrics.csv"
	with csv_path.open("w", newline="", encoding="utf-8") as f:
		writer = csv.writer(f)
		writer.writerow(["episode", "objective_cost_real_power", "avg_delay_per_user"])
		for episode, objective, delay in zip(episodes, objective_cost.reshape(-1), avg_delay.reshape(-1)):
			writer.writerow([int(episode), float(objective), float(delay)])

	fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), dpi=160)
	axes[0].plot(episodes, objective_cost.reshape(-1), color="#1f77b4", linewidth=1.8)
	axes[0].set_title("Objective cost: real sum(power)")
	axes[0].set_xlabel("episode")
	axes[0].set_ylabel("real power")
	axes[0].grid(True, alpha=0.25)
	axes[1].plot(episodes, avg_delay.reshape(-1), color="#d62728", linewidth=1.8)
	axes[1].axhline(1.2, color="#555555", linestyle="--", linewidth=1.0)
	axes[1].set_title("Average delay per user")
	axes[1].set_xlabel("episode")
	axes[1].set_ylabel("delay")
	axes[1].grid(True, alpha=0.25)
	fig.suptitle("SLDAC_code MIMO dual, seed=0, B=100, q=1, 100 episodes")
	fig.tight_layout(rect=(0, 0, 1, 0.94))
	png_path = OUT_DIR / "SLDAC_dual_seed0_ep100_curves.png"
	pdf_path = OUT_DIR / "SLDAC_dual_seed0_ep100_curves.pdf"
	fig.savefig(png_path)
	fig.savefig(pdf_path)
	plt.close(fig)

	summary = {
		"source": str(MIMO_DIR),
		"seed": SEED,
		"solver": "lagrangian_dual",
		"config": {
			"T": T,
			"num_new_data": NUM_NEW_DATA,
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
		},
		"num_points": int(objective_cost.size),
		"final_objective_cost_real_power": float(objective_cost.reshape(-1)[-1]),
		"final_avg_delay_per_user": float(avg_delay.reshape(-1)[-1]),
		"mean_last_10_objective_cost_real_power": float(np.mean(objective_cost.reshape(-1)[-10:])),
		"mean_last_10_avg_delay_per_user": float(np.mean(avg_delay.reshape(-1)[-10:])),
		"files": {
			"reward_mat": str(reward_mat),
			"cost_mat": str(cost_mat),
			"csv": str(csv_path),
			"png": str(png_path),
			"pdf": str(pdf_path),
		},
	}
	summary_path = OUT_DIR / "summary.json"
	summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
	print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


def main():
	import utils

	utils.update_policy = dual_update_policy
	from SLDAC import SLDAC_main

	objective_cost, avg_delay = SLDAC_main(build_args(), "MIMO")
	write_outputs(objective_cost, avg_delay)


if __name__ == "__main__":
	main()
