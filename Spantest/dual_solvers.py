from __future__ import annotations

import time

import numpy as np
from scipy.optimize import minimize

from .cssca_solvers import CsscaProblem, CsscaResult, row_space_basis


def _surrogate_values(values: np.ndarray, gradients: np.ndarray, tau: np.ndarray, step: np.ndarray) -> np.ndarray:
    return values + gradients @ step + tau * float(step @ step)


def _status(success: bool) -> str:
    return "optimal" if success else "optimal_inaccurate"


def _solve_feasible_dual(values: np.ndarray, gradients: np.ndarray, tau: np.ndarray, max_iter: int):
    m = int(values.shape[0])
    gram = gradients @ gradients.T

    def dual_value(lam: np.ndarray) -> float:
        tau_lam = float(tau @ lam)
        quad = float(lam @ gram @ lam)
        return float(values @ lam - quad / (4.0 * tau_lam))

    def objective(lam: np.ndarray) -> float:
        return -dual_value(lam)

    def jacobian(lam: np.ndarray) -> np.ndarray:
        tau_lam = float(tau @ lam)
        gram_lam = gram @ lam
        quad = float(lam @ gram_lam)
        grad = values - ((2.0 * gram_lam * tau_lam - quad * tau) / (4.0 * tau_lam * tau_lam))
        return -grad

    result = minimize(
        objective,
        np.full(m, 1.0 / m, dtype=np.float64),
        jac=jacobian,
        method="SLSQP",
        bounds=[(0.0, None)] * m,
        constraints=({"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0), "jac": lambda x: np.ones_like(x)}),
        options={"ftol": 1e-10, "maxiter": int(max_iter), "disp": False},
    )
    lam = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    lam_sum = float(np.sum(lam))
    if lam_sum <= 0.0:
        lam = np.full(m, 1.0 / m, dtype=np.float64)
    else:
        lam = lam / lam_sum
    tau_lam = float(tau @ lam)
    step = -(lam @ gradients) / (2.0 * tau_lam)
    return step, lam, bool(result.success)


def _solve_objective_dual(values: np.ndarray, gradients: np.ndarray, tau: np.ndarray, max_iter: int):
    constraint_values = values[1:]
    constraint_gradients = gradients[1:]
    constraint_tau = tau[1:]
    g0 = gradients[0]
    gram_cc = constraint_gradients @ constraint_gradients.T
    gram_0c = constraint_gradients @ g0
    gram_00 = float(g0 @ g0)
    tau0 = float(tau[0])
    value0 = float(values[0])
    m = int(constraint_values.shape[0])

    def dual_value(lam: np.ndarray) -> float:
        tau_lam = tau0 + float(constraint_tau @ lam)
        combo_quad = gram_00 + 2.0 * float(gram_0c @ lam) + float(lam @ gram_cc @ lam)
        return float(value0 + constraint_values @ lam - combo_quad / (4.0 * tau_lam))

    def objective(lam: np.ndarray) -> float:
        return -dual_value(lam)

    def jacobian(lam: np.ndarray) -> np.ndarray:
        tau_lam = tau0 + float(constraint_tau @ lam)
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
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": int(max_iter), "maxls": 50},
    )
    lam = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    tau_lam = tau0 + float(constraint_tau @ lam)
    step = -(g0 + lam @ constraint_gradients) / (2.0 * tau_lam)
    return step, lam, bool(result.success)


def solve_cssca_dual(problem: CsscaProblem, feasibility_tol: float = 1e-8, max_iter: int = 500) -> CsscaResult:
    values = np.asarray(problem.values, dtype=np.float64)
    gradients = np.asarray(problem.gradients, dtype=np.float64)
    theta = np.asarray(problem.theta, dtype=np.float64)
    tau = np.asarray(problem.tau, dtype=np.float64)
    t0 = time.perf_counter()

    feasible_step, feasible_lambda, feasible_success = _solve_feasible_dual(
        values[1:], gradients[1:], tau[1:], max_iter=max_iter
    )
    feasible_surrogates = _surrogate_values(values, gradients, tau, feasible_step)
    feasible_x = float(np.max(feasible_surrogates[1:]))

    if feasible_x <= feasibility_tol:
        step, objective_lambda, objective_success = _solve_objective_dual(values, gradients, tau, max_iter=max_iter)
        branch = "objective"
        status = _status(objective_success)
        surrogates = _surrogate_values(values, gradients, tau, step)
        objective_value = float(surrogates[0])
        active_rows = [idx + 1 for idx, item in enumerate(objective_lambda) if item > 1e-7]
    else:
        step = feasible_step
        branch = "feasible"
        status = _status(feasible_success)
        objective_value = feasible_x
        active_rows = [idx + 1 for idx, item in enumerate(feasible_lambda) if item > 1e-7]

    elapsed = time.perf_counter() - t0
    return CsscaResult(
        theta_bar=theta + step,
        step=step,
        status=status,
        branch=branch,
        objective_value=objective_value,
        feasible_x=feasible_x,
        solve_time_sec=float(elapsed),
        decision_dim=int(problem.constraint_dim),
        gradient_rank=int(row_space_basis(gradients).shape[1]),
        active_rows=active_rows,
        solver="dual-scipy",
    )
