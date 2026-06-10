from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize


@dataclass
class CsscaResult:
    theta_bar: np.ndarray
    status: str
    success: bool
    step_norm: float
    message: str


def _nonfinite_result(theta, status):
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    return CsscaResult(theta_bar=theta.copy(), status=status, success=False, step_norm=0.0, message=status)


def _as_arrays(func_value_np, grad_np, theta_np):
    values = np.asarray(func_value_np, dtype=np.float64).reshape(-1)
    gradients = np.asarray(grad_np, dtype=np.float64)
    theta = np.asarray(theta_np, dtype=np.float64).reshape(-1)
    if gradients.ndim != 2 or gradients.shape[0] != values.size or gradients.shape[1] != theta.size:
        raise ValueError("CSSCA shape mismatch")
    return values, gradients, theta


def _feasible_candidate(values, gradients, tau_cost):
    constraint_values = values[1:]
    constraint_gradients = gradients[1:]
    m = constraint_values.size
    if m == 0:
        return np.zeros(gradients.shape[1], dtype=np.float64), -np.inf, True, "no_constraints"
    tau = float(tau_cost)
    if m == 1:
        lam = np.ones(1, dtype=np.float64)
        denom = 2.0 * tau
        step = -constraint_gradients[0] / denom
        value = constraint_values[0] + constraint_gradients[0].dot(step) + tau * step.dot(step)
        return step, value, True, "feasible_single_constraint"

    def objective(lam):
        weighted_grad = lam @ constraint_gradients
        weighted_tau = max(float(np.sum(lam) * tau), 1e-12)
        step = -weighted_grad / (2.0 * weighted_tau)
        dual_value = float(np.dot(lam, constraint_values) + weighted_grad.dot(step) + weighted_tau * step.dot(step))
        return -dual_value

    cons = [{"type": "eq", "fun": lambda lam: float(np.sum(lam) - 1.0)}]
    bounds = [(0.0, None) for _ in range(m)]
    init = np.ones(m, dtype=np.float64) / m
    result = minimize(objective, init, method="SLSQP", bounds=bounds, constraints=cons, options={"maxiter": 200, "ftol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        return np.zeros(gradients.shape[1], dtype=np.float64), np.inf, False, "feasible_dual_failed"
    lam = np.maximum(result.x, 0.0)
    lam = lam / max(float(np.sum(lam)), 1e-12)
    weighted_grad = lam @ constraint_gradients
    weighted_tau = max(float(np.sum(lam) * tau), 1e-12)
    step = -weighted_grad / (2.0 * weighted_tau)
    max_value = np.max(constraint_values + constraint_gradients @ step + tau * step.dot(step))
    return step, float(max_value), True, "feasible_dual_success"


def _objective_candidate(values, gradients, tau_reward, tau_cost):
    objective_value = values[0]
    objective_grad = gradients[0]
    constraint_values = values[1:]
    constraint_gradients = gradients[1:]
    m = constraint_values.size
    if m == 0:
        step = -objective_grad / (2.0 * float(tau_reward))
        return step, True, "objective_unconstrained"

    tau0 = float(tau_reward)
    tauc = float(tau_cost)

    def dual_objective(lam):
        weighted_grad = objective_grad + lam @ constraint_gradients
        weighted_tau = tau0 + float(np.sum(lam)) * tauc
        step = -weighted_grad / (2.0 * weighted_tau)
        dual_value = (
            objective_value
            + np.dot(lam, constraint_values)
            + weighted_grad.dot(step)
            + weighted_tau * step.dot(step)
        )
        return -float(dual_value)

    bounds = [(0.0, None) for _ in range(m)]
    init = np.zeros(m, dtype=np.float64)
    result = minimize(dual_objective, init, method="SLSQP", bounds=bounds, options={"maxiter": 200, "ftol": 1e-9})
    if not result.success or not np.isfinite(result.x).all():
        return np.zeros(gradients.shape[1], dtype=np.float64), False, "objective_dual_failed"
    lam = np.maximum(result.x, 0.0)
    weighted_grad = objective_grad + lam @ constraint_gradients
    weighted_tau = tau0 + float(np.sum(lam)) * tauc
    step = -weighted_grad / (2.0 * weighted_tau)
    return step, True, "objective_dual_success"


def update_policy(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost, return_info=False):
    try:
        values, gradients, theta = _as_arrays(func_value_np, grad_np, paras_t_np)
    except ValueError:
        result = _nonfinite_result(paras_t_np, "fallback_shape_mismatch")
        return result if return_info else result.theta_bar
    if (not np.isfinite(values).all()) or (not np.isfinite(gradients).all()) or (not np.isfinite(theta).all()):
        result = _nonfinite_result(theta, "fallback_nonfinite_input")
        return result if return_info else result.theta_bar

    feasible_step, feasible_value, feasible_success, feasible_status = _feasible_candidate(values, gradients, tau_cost)
    if not feasible_success:
        theta_bar = theta + feasible_step
        result = CsscaResult(theta_bar=theta_bar, status=feasible_status, success=False, step_norm=float(np.linalg.norm(feasible_step)), message=feasible_status)
        return result if return_info else result.theta_bar

    if feasible_value <= 0:
        step, success, status = _objective_candidate(values, gradients, tau_reward, tau_cost)
    else:
        step, success, status = feasible_step, True, "feasible_update"

    theta_bar = theta + step
    if (not success) or (not np.isfinite(theta_bar).all()):
        result = CsscaResult(theta_bar=theta.copy(), status=status, success=False, step_norm=0.0, message=status)
    else:
        result = CsscaResult(theta_bar=theta_bar, status=status, success=True, step_norm=float(np.linalg.norm(step)), message=status)
    return result if return_info else result.theta_bar

