import numpy as np
import unittest

from Spantest.cssca_solvers import (
    CsscaProblem,
    solve_active_span_cvxpy,
    solve_cssca,
    solve_full_cvxpy,
    solve_gradient_span_cvxpy,
)


def _unit(vector):
    return vector / np.linalg.norm(vector)


def make_objective_problem(n_theta=80):
    rng = np.random.default_rng(123)
    g0 = _unit(rng.normal(size=n_theta))
    g1 = -g0
    g2 = _unit(rng.normal(size=n_theta))
    g2 = _unit(g2 - g0 * float(g2 @ g0))
    gradients = np.vstack([g0, g1, g2])
    values = np.array([0.0, 0.10, -2.0], dtype=np.float64)
    tau = np.ones(3, dtype=np.float64)
    theta = rng.normal(scale=0.01, size=n_theta)
    return CsscaProblem(values=values, gradients=gradients, theta=theta, tau=tau)


class CsscaSolverTests(unittest.TestCase):
    def test_gradient_span_matches_full_cvxpy_solution(self):
        problem = make_objective_problem()

        full = solve_full_cvxpy(problem)
        span = solve_gradient_span_cvxpy(problem)

        self.assertTrue(full.status.startswith("optimal"))
        self.assertTrue(span.status.startswith("optimal"))
        self.assertEqual(full.branch, "objective")
        self.assertEqual(span.branch, "objective")
        self.assertLess(np.linalg.norm(full.step - span.step) / np.linalg.norm(full.step), 5e-4)
        self.assertLess(abs(full.objective_value - span.objective_value), 5e-4)
        self.assertLess(span.decision_dim, problem.theta_dim)

    def test_active_span_can_reduce_objective_case_to_one_dimension(self):
        problem = make_objective_problem()

        active = solve_active_span_cvxpy(problem)

        self.assertTrue(active.status.startswith("optimal"))
        self.assertEqual(active.branch, "objective")
        self.assertEqual(active.decision_dim, 1)
        self.assertEqual(active.active_rows, [0, 1])

    def test_dual_solver_is_available_through_dispatch(self):
        problem = make_objective_problem()

        dual = solve_cssca(problem, "dual")
        span = solve_gradient_span_cvxpy(problem)

        self.assertTrue(dual.status.startswith("optimal"))
        self.assertEqual(dual.branch, span.branch)
        self.assertLess(np.linalg.norm(dual.step - span.step), 1e-4)


if __name__ == "__main__":
    unittest.main()
