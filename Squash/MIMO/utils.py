import cvxpy as cp
import numpy as np
import torch
import shutil

from cssca_dual_solver import CSSCA_SOLVER_DUAL, normalize_cssca_solver, solve_dual_cssca_update

# CVXPY/MOSEK 求解配置：优先使用 MOSEK，失败后自动回退到其他可用求解器。
SOLVER_PRIORITY = ("MOSEK", "OSQP", "ECOS", "SCS", "CLARABEL", "SCIPY")
# 参考 CVXPY 官方 MOSEK 说明：连续问题会被 dualize，显式指定 dual form 更稳妥。
DEFAULT_MOSEK_PARAMS = {
	"MSK_IPAR_INTPNT_SOLVE_FORM": "MSK_SOLVE_DUAL",
}

def soft_update_twoloop(target, source, tau, Q_update_time, Q_update_index):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	if Q_update_index==1:
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau/Q_update_time)
	else:
		for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(target_param.data + param.data * tau/Q_update_time)

def soft_update(target, source, tau):
	"""
	Copies the parameters from source network (x) to target network (y) using the below update
	y = TAU*x + (1 - TAU)*y
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
		target_param.data.copy_(
			target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
	"""
	Copies the parameters from source network to target network
	:param target: Target network (PyTorch)
	:param source: Source network (PyTorch)
	:return:
	"""
	for target_param, param in zip(target.parameters(), source.parameters()):
			target_param.data.copy_(param.data)


def _build_solver_candidates():
	installed = set(cp.installed_solvers())
	candidates = []
	for solver_name in SOLVER_PRIORITY:
		if (solver_name not in installed) or (not hasattr(cp, solver_name)):
			continue
		solver = getattr(cp, solver_name)
		solver_kwargs = {"warm_start": True}
		if solver_name == "MOSEK":
			solver_kwargs["mosek_params"] = dict(DEFAULT_MOSEK_PARAMS)
		candidates.append((solver, solver_kwargs))
	return candidates


def _solve_problem(prob):
	last_err = None
	for solver, solver_kwargs in _build_solver_candidates():
		try:
			prob.solve(solver=solver, **solver_kwargs)
		except Exception as ex:
			last_err = ex
			continue
		if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
			return prob.status
	if last_err is not None:
		print("cvxpy fallback failed:", repr(last_err))
	prob.solve(warm_start=True)
	return prob.status




def update_policy(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost, cssca_solver="cvx"):
	cssca_solver = normalize_cssca_solver(cssca_solver)
	if cssca_solver == CSSCA_SOLVER_DUAL:
		try:
			paras_bar, info = solve_dual_cssca_update(
				func_value_np,
				grad_np,
				paras_t_np,
				tau_reward=tau_reward,
				tau_cost=tau_cost,
			)
			if paras_bar is not None and np.isfinite(paras_bar).all():
				return paras_bar
			print("dual CSSCA fallback to cvx: status =", None if info is None else info.get("status"))
		except Exception as ex:
			print("dual CSSCA fallback to cvx:", repr(ex))

	x, paras_bar, prob_status_fea = _feasible_update(func_value_np, grad_np, paras_t_np, tau_cost)
	if x == np.inf:
		print('feasible problem break ! status = ', prob_status_fea)

	if x <= 0:
		paras_bar, prob_status_obj = _objective_update(func_value_np, grad_np, paras_t_np,tau_reward=tau_reward, tau_cost=tau_cost)
		if paras_bar is None:
			print('objective problem break ! status = ', prob_status_obj)

	return paras_bar


def _objective_update(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost):
	m = grad_np.shape[0] - 1  # number of constraints.
	n = grad_np.shape[1]  # dim of parameter.
	tau_np = tau_cost * np.ones(m + 1)
	tau_np[0] = tau_reward

	paras_cvx = cp.Variable(shape=(n,))
	obj = func_value_np[0] + grad_np[0].T @ (paras_cvx - paras_t_np) + tau_np[0] * cp.sum_squares(paras_cvx - paras_t_np)
	constr = []
	for i in range(1, m + 1):
		constr += [func_value_np[i] + grad_np[i].T @ (paras_cvx - paras_t_np) + tau_np[i] * cp.sum_squares(paras_cvx - paras_t_np) <= 0]
	prob = cp.Problem(cp.Minimize(obj), constr)
	_solve_problem(prob)
	paras_mosek = paras_cvx.value

	return paras_mosek, prob.status


def _feasible_update(func_value_np, grad_np, paras_t_np, tau_cost):
	m = grad_np.shape[0] - 1  # number of constraints.
	n = grad_np.shape[1]  # dim of parameter.
	func_value_np = func_value_np[1:]
	grad_np = grad_np[1:]
	tau_np = tau_cost * np.ones(m)

	paras_cvx = cp.Variable(shape=(n,))
	x_cvx = cp.Variable()
	obj = x_cvx
	constr = []
	for i in range(m):
		constr += [func_value_np[i] + grad_np[i].T @ (paras_cvx - paras_t_np) + tau_np[i] * cp.sum_squares(paras_cvx - paras_t_np) <= x_cvx]
	prob = cp.Problem(cp.Minimize(obj), constr)
	_solve_problem(prob)
	x_mosek = prob.value
	paras_mosek = paras_cvx.value

	return x_mosek, paras_mosek, prob.status
