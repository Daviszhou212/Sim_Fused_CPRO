import numpy as np
from scipy.optimize import minimize


CSSCA_SOLVER_CVX = "cvx"
CSSCA_SOLVER_DUAL = "dual"
CSSCA_SOLVER_CHOICES = (CSSCA_SOLVER_CVX, CSSCA_SOLVER_DUAL)


def normalize_cssca_solver(value):
    if value is None:
        return CSSCA_SOLVER_CVX
    text = str(value).strip().lower().replace("-", "_")
    if text in ("cvx", "cvxpy", "mosek"):
        return CSSCA_SOLVER_CVX
    if text in ("dual", "lagrangian", "lagrange", "dual_scipy"):
        return CSSCA_SOLVER_DUAL
    raise ValueError("unsupported cssca_solver: {0!r}".format(value))


def can_use_dual_cssca(simplex_dim=None):
    return simplex_dim is None or int(simplex_dim) <= 0


def _as_problem_arrays(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost):
    values = np.asarray(func_value_np, dtype=np.float64).reshape(-1)
    gradients = np.asarray(grad_np, dtype=np.float64)
    theta = np.asarray(paras_t_np, dtype=np.float64).reshape(-1)
    if gradients.ndim != 2:
        raise ValueError("grad_np must be a 2-D array")
    if values.size != gradients.shape[0]:
        raise ValueError(
            "func_value_np and grad_np row mismatch: {0} != {1}".format(
                values.size,
                gradients.shape[0],
            )
        )
    if theta.size != gradients.shape[1]:
        raise ValueError(
            "paras_t_np and grad_np dimension mismatch: {0} != {1}".format(
                theta.size,
                gradients.shape[1],
            )
        )
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


def _solve_feasible_dual(values, gradients, tau, max_iter):
    m = int(values.shape[0])
    if m <= 0:
        return np.zeros(gradients.shape[1], dtype=np.float64), np.zeros(0, dtype=np.float64), True
    if m == 1:
        lam = np.ones(1, dtype=np.float64)
        step = -gradients[0] / (2.0 * float(tau[0]))
        return step, lam, True

    gram = gradients @ gradients.T

    def objective(lam):
        tau_lam = float(tau @ lam)
        quad = float(lam @ gram @ lam)
        return -float(values @ lam - quad / (4.0 * tau_lam))

    def jacobian(lam):
        tau_lam = float(tau @ lam)
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
        options={"ftol": 1e-10, "maxiter": int(max_iter), "disp": False},
    )
    lam = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    lam_sum = float(np.sum(lam))
    if lam_sum <= 0.0:
        lam = np.full(m, 1.0 / float(m), dtype=np.float64)
    else:
        lam = lam / lam_sum
    tau_lam = float(tau @ lam)
    step = -(lam @ gradients) / (2.0 * tau_lam)
    return step, lam, bool(result.success)


def _solve_objective_dual(values, gradients, tau, max_iter):
    g0 = gradients[0]
    tau0 = float(tau[0])
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
        tau_lam = tau0 + float(constraint_tau @ lam)
        combo_quad = gram_00 + 2.0 * float(gram_0c @ lam) + float(lam @ gram_cc @ lam)
        return -float(value0 + constraint_values @ lam - combo_quad / (4.0 * tau_lam))

    def jacobian(lam):
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


def solve_dual_cssca_update(
    func_value_np,
    grad_np,
    paras_t_np,
    tau_reward,
    tau_cost,
    feasibility_tol=1e-8,
    max_iter=500,
):
    values, gradients, theta, tau = _as_problem_arrays(
        func_value_np,
        grad_np,
        paras_t_np,
        tau_reward,
        tau_cost,
    )

    feasible_step, feasible_lambda, feasible_success = _solve_feasible_dual(
        values[1:],
        gradients[1:],
        tau[1:],
        max_iter=max_iter,
    )
    feasible_surrogates = _surrogate_values(values, gradients, tau, feasible_step)
    feasible_x = -np.inf if values.size == 1 else float(np.max(feasible_surrogates[1:]))

    if feasible_x <= float(feasibility_tol):
        step, lam, success = _solve_objective_dual(values, gradients, tau, max_iter=max_iter)
        surrogates = _surrogate_values(values, gradients, tau, step)
        info = {
            "solver": CSSCA_SOLVER_DUAL,
            "branch": "objective",
            "status": "optimal" if success else "optimal_inaccurate",
            "objective_value": float(surrogates[0]),
            "feasible_x": feasible_x,
            "lambda": lam,
        }
    else:
        step = feasible_step
        info = {
            "solver": CSSCA_SOLVER_DUAL,
            "branch": "feasible",
            "status": "optimal" if feasible_success else "optimal_inaccurate",
            "objective_value": feasible_x,
            "feasible_x": feasible_x,
            "lambda": feasible_lambda,
        }
    return theta + step, info
