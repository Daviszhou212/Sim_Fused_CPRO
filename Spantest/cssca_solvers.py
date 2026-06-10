from __future__ import annotations

from dataclasses import dataclass, field
import time

import cvxpy as cp
import numpy as np


SOLVER_PRIORITY = ("CLARABEL", "MOSEK", "SCS", "SCIPY")


@dataclass(frozen=True)
class CsscaProblem:
    values: np.ndarray
    gradients: np.ndarray
    theta: np.ndarray
    tau: np.ndarray

    @property
    def theta_dim(self) -> int:
        return int(self.gradients.shape[1])

    @property
    def constraint_dim(self) -> int:
        return int(self.gradients.shape[0] - 1)


@dataclass
class CsscaResult:
    theta_bar: np.ndarray
    step: np.ndarray
    status: str
    branch: str
    objective_value: float
    feasible_x: float
    solve_time_sec: float
    decision_dim: int
    gradient_rank: int
    active_rows: list[int] = field(default_factory=list)
    solver: str = ""


def row_space_basis(rows: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    rows = np.asarray(rows, dtype=np.float64)
    if rows.size == 0:
        return np.zeros((rows.shape[1], 0), dtype=np.float64)
    _, singular_values, v_t = np.linalg.svd(rows, full_matrices=False)
    if singular_values.size == 0:
        return np.zeros((rows.shape[1], 0), dtype=np.float64)
    cutoff = tol * max(rows.shape) * max(float(singular_values[0]), 1.0)
    rank = int(np.sum(singular_values > cutoff))
    return v_t[:rank].T.copy()


def _solve_problem(problem: cp.Problem):
    installed = set(cp.installed_solvers())
    last_error = None
    for solver_name in SOLVER_PRIORITY:
        if solver_name not in installed:
            continue
        try:
            if solver_name == "CLARABEL":
                problem.solve(
                    solver=getattr(cp, solver_name),
                    tol_gap_abs=1e-8,
                    tol_gap_rel=1e-8,
                    tol_feas=1e-8,
                    max_iter=1000,
                    verbose=False,
                )
            elif solver_name == "SCS":
                problem.solve(solver=getattr(cp, solver_name), eps=1e-6, max_iters=20000, verbose=False)
            else:
                problem.solve(solver=getattr(cp, solver_name), verbose=False)
        except Exception as ex:  # pragma: no cover - diagnostic fallback
            last_error = ex
            continue
        if problem.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return solver_name
    raise RuntimeError(f"no cvxpy solver succeeded; last_error={last_error!r}")


def _surrogate_values(values: np.ndarray, gradients: np.ndarray, tau: np.ndarray, step: np.ndarray) -> np.ndarray:
    return values + gradients @ step + tau * float(step @ step)


def _solve_subproblem(problem: CsscaProblem, basis: np.ndarray | None):
    values = np.asarray(problem.values, dtype=np.float64)
    gradients = np.asarray(problem.gradients, dtype=np.float64)
    theta = np.asarray(problem.theta, dtype=np.float64)
    tau = np.asarray(problem.tau, dtype=np.float64)
    if basis is None:
        decision = cp.Variable(problem.theta_dim)
        step_expr = decision - theta
        linear_gradients = gradients
        decision_dim = problem.theta_dim
    else:
        decision = cp.Variable(basis.shape[1])
        step_expr = decision
        linear_gradients = gradients @ basis
        decision_dim = int(basis.shape[1])

    t0 = time.perf_counter()
    step_norm = cp.sum_squares(step_expr)
    x_var = cp.Variable()
    feasible_constraints = [
        values[i] + linear_gradients[i] @ step_expr + tau[i] * step_norm <= x_var
        for i in range(1, gradients.shape[0])
    ]
    feasible_problem = cp.Problem(cp.Minimize(x_var), feasible_constraints)
    solver_name = _solve_problem(feasible_problem)
    feasible_x = float(feasible_problem.value)

    if feasible_x <= 0.0:
        objective = values[0] + linear_gradients[0] @ step_expr + tau[0] * step_norm
        objective_constraints = [
            values[i] + linear_gradients[i] @ step_expr + tau[i] * step_norm <= 0.0
            for i in range(1, gradients.shape[0])
        ]
        objective_problem = cp.Problem(cp.Minimize(objective), objective_constraints)
        solver_name = _solve_problem(objective_problem)
        status = str(objective_problem.status)
        objective_value = float(objective_problem.value)
        branch = "objective"
    else:
        status = str(feasible_problem.status)
        objective_value = feasible_x
        branch = "feasible"

    raw_step = np.asarray(decision.value, dtype=np.float64).reshape(-1)
    step = raw_step - theta if basis is None else basis @ raw_step
    theta_bar = theta + step
    elapsed = time.perf_counter() - t0
    return CsscaResult(
        theta_bar=theta_bar,
        step=step,
        status=status,
        branch=branch,
        objective_value=objective_value,
        feasible_x=feasible_x,
        solve_time_sec=float(elapsed),
        decision_dim=decision_dim,
        gradient_rank=decision_dim if basis is not None else int(row_space_basis(gradients).shape[1]),
        solver=solver_name,
    )


def solve_full_cvxpy(problem: CsscaProblem) -> CsscaResult:
    return _solve_subproblem(problem, basis=None)


def solve_gradient_span_cvxpy(problem: CsscaProblem) -> CsscaResult:
    basis = row_space_basis(problem.gradients)
    return _solve_subproblem(problem, basis=basis)


def solve_active_span_cvxpy(problem: CsscaProblem) -> CsscaResult:
    full_basis = row_space_basis(problem.gradients)
    prelim = _solve_subproblem(problem, basis=full_basis)
    surrogate = _surrogate_values(problem.values, problem.gradients, problem.tau, prelim.step)
    if prelim.branch == "objective":
        active_rows = [0] + [i for i in range(1, problem.gradients.shape[0]) if surrogate[i] >= -1e-6]
    else:
        active_rows = [
            i for i in range(1, problem.gradients.shape[0]) if surrogate[i] >= prelim.feasible_x - 1e-6
        ]
        if not active_rows:
            active_rows = list(range(1, problem.gradients.shape[0]))
    basis = row_space_basis(problem.gradients[active_rows])
    result = _solve_subproblem(problem, basis=basis)
    result.active_rows = active_rows
    return result


def solve_cssca(problem: CsscaProblem, solver_name: str) -> CsscaResult:
    if solver_name == "full":
        return solve_full_cvxpy(problem)
    if solver_name == "span":
        return solve_gradient_span_cvxpy(problem)
    if solver_name == "active":
        return solve_active_span_cvxpy(problem)
    if solver_name == "dual":
        from .dual_solvers import solve_cssca_dual

        return solve_cssca_dual(problem)
    raise ValueError(f"unknown solver_name {solver_name!r}")
