import numpy as np
from scipy.optimize import minimize


def _as_problem_arrays(func_value, grad, theta, tau_objective, tau_constraint):
    values = np.asarray(func_value, dtype=np.float64).reshape(-1)
    gradients = np.asarray(grad, dtype=np.float64)
    theta = np.asarray(theta, dtype=np.float64).reshape(-1)
    if gradients.ndim != 2:
        raise ValueError("grad must be a 2-D array")
    if values.size != gradients.shape[0]:
        raise ValueError("func_value and grad row mismatch")
    if theta.size != gradients.shape[1]:
        raise ValueError("theta and grad dimension mismatch")
    if values.size < 1:
        raise ValueError("CSSCA problem needs an objective")
    if float(tau_objective) <= 0.0 or float(tau_constraint) <= 0.0:
        raise ValueError("tau values must be positive")
    if not (np.isfinite(values).all() and np.isfinite(gradients).all() and np.isfinite(theta).all()):
        raise ValueError("CSSCA inputs must be finite")
    tau = float(tau_constraint) * np.ones(values.size, dtype=np.float64)
    tau[0] = float(tau_objective)
    return values, gradients, theta, tau


def _surrogate(values, gradients, tau, step):
    return values + gradients @ step + tau * float(step @ step)


def _solve_feasible_dual(values, gradients, tau, max_iter):
    m = int(values.size)
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
    lam = np.full(m, 1.0 / float(m), dtype=np.float64) if lam_sum <= 0.0 else lam / lam_sum
    step = -(lam @ gradients) / (2.0 * float(tau @ lam))
    return step, lam, bool(result.success)


def _solve_objective_dual(values, gradients, tau, max_iter):
    g0 = gradients[0]
    tau0 = float(tau[0])
    constraint_count = int(values.size - 1)
    if constraint_count <= 0:
        return -g0 / (2.0 * tau0), np.zeros(0, dtype=np.float64), True

    c_values = values[1:]
    c_grad = gradients[1:]
    c_tau = tau[1:]
    gram_cc = c_grad @ c_grad.T
    gram_0c = c_grad @ g0
    gram_00 = float(g0 @ g0)

    def objective(lam):
        tau_lam = tau0 + float(c_tau @ lam)
        combo_quad = gram_00 + 2.0 * float(gram_0c @ lam) + float(lam @ gram_cc @ lam)
        return -float(values[0] + c_values @ lam - combo_quad / (4.0 * tau_lam))

    def jacobian(lam):
        tau_lam = tau0 + float(c_tau @ lam)
        gram_lam = gram_0c + gram_cc @ lam
        combo_quad = gram_00 + 2.0 * float(gram_0c @ lam) + float(lam @ gram_cc @ lam)
        grad = c_values - ((2.0 * gram_lam * tau_lam - combo_quad * c_tau) / (4.0 * tau_lam * tau_lam))
        return -grad

    result = minimize(
        objective,
        np.zeros(constraint_count, dtype=np.float64),
        jac=jacobian,
        method="L-BFGS-B",
        bounds=[(0.0, None)] * constraint_count,
        options={"ftol": 1e-12, "gtol": 1e-9, "maxiter": int(max_iter), "maxls": 50},
    )
    lam = np.maximum(np.asarray(result.x, dtype=np.float64), 0.0)
    step = -(g0 + lam @ c_grad) / (2.0 * (tau0 + float(c_tau @ lam)))
    return step, lam, bool(result.success)


def _solve_lagrangian_dual(values, gradients, theta, tau, feasibility_tol, max_iter):
    feasible_step, feasible_lambda, feasible_success = _solve_feasible_dual(
        values[1:],
        gradients[1:],
        tau[1:],
        max_iter=max_iter,
    )
    feasible_surrogates = _surrogate(values, gradients, tau, feasible_step)
    feasible_x = -np.inf if values.size == 1 else float(np.max(feasible_surrogates[1:]))
    if feasible_x <= float(feasibility_tol):
        step, lam, success = _solve_objective_dual(values, gradients, tau, max_iter=max_iter)
        branch = "objective"
        surrogate_values = _surrogate(values, gradients, tau, step)
    else:
        step = feasible_step
        lam = feasible_lambda
        success = feasible_success
        branch = "feasible"
        surrogate_values = feasible_surrogates
    return theta + step, {
        "solver": "lagrangian_dual",
        "branch": branch,
        "status": "optimal" if success else "optimal_inaccurate",
        "lambda": lam,
        "feasible_x": feasible_x,
        "surrogate": surrogate_values,
    }


def _solve_cvxpy(values, gradients, theta, tau):
    try:
        import cvxpy as cp
    except Exception as exc:
        raise RuntimeError("cvxpy fallback unavailable") from exc

    variable = cp.Variable(theta.size)
    diff = variable - theta
    constraints = []
    for idx in range(1, values.size):
        constraints.append(values[idx] + gradients[idx].T @ diff + tau[idx] * cp.sum_squares(diff) <= 0)
    objective = values[0] + gradients[0].T @ diff + tau[0] * cp.sum_squares(diff)
    problem = cp.Problem(cp.Minimize(objective), constraints)
    problem.solve(warm_start=True)
    if variable.value is None:
        raise RuntimeError("cvxpy returned no solution")
    return np.asarray(variable.value, dtype=np.float64).reshape(-1), {
        "solver": "cvx",
        "branch": "objective",
        "status": str(problem.status),
    }


def solve_cssca_update(
    func_value,
    grad,
    theta,
    tau_objective,
    tau_constraint,
    solver="lagrangian_dual",
    feasibility_tol=1e-8,
    max_iter=500,
):
    values, gradients, theta, tau = _as_problem_arrays(func_value, grad, theta, tau_objective, tau_constraint)
    solver = str(solver)
    if solver == "lagrangian_dual":
        next_theta, info = _solve_lagrangian_dual(values, gradients, theta, tau, feasibility_tol, max_iter)
    elif solver == "cvx":
        next_theta, info = _solve_cvxpy(values, gradients, theta, tau)
    else:
        raise ValueError("unsupported CSSCA solver: {0}".format(solver))

    next_theta = np.asarray(next_theta, dtype=np.float64).reshape(-1)
    if next_theta.shape != theta.shape:
        raise ValueError("solver returned shape mismatch")
    if not np.isfinite(next_theta).all():
        raise ValueError("solver returned non-finite theta")
    return next_theta, info
