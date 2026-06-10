import unittest

import numpy as np

from Spantest.cssca_solvers import CsscaProblem, row_space_basis, solve_gradient_span_cvxpy
from Spantest.dual_solvers import solve_cssca_dual


class DualSolverTests(unittest.TestCase):
    def test_dual_objective_matches_span_cvxpy_on_small_problem(self):
        rng = np.random.default_rng(7)
        gradients = rng.normal(size=(5, 16))
        values = np.array([0.2, -1.0, -1.1, -1.2, -1.3], dtype=np.float64)
        tau = np.ones(5, dtype=np.float64)
        theta = rng.normal(size=16)
        problem = CsscaProblem(values=values, gradients=gradients, theta=theta, tau=tau)

        span_result = solve_gradient_span_cvxpy(problem)
        dual_result = solve_cssca_dual(problem)

        self.assertEqual(dual_result.branch, "objective")
        self.assertLess(np.linalg.norm(span_result.step - dual_result.step), 1e-4)
        self.assertAlmostEqual(span_result.objective_value, dual_result.objective_value, places=4)

    def test_dual_step_lies_in_gradient_span(self):
        rng = np.random.default_rng(11)
        gradients = rng.normal(size=(5, 64))
        values = np.array([0.0, 100.0, 100.0, 100.0, 100.0], dtype=np.float64)
        tau = np.ones(5, dtype=np.float64)
        theta = np.zeros(64, dtype=np.float64)
        problem = CsscaProblem(values=values, gradients=gradients, theta=theta, tau=tau)

        result = solve_cssca_dual(problem)
        basis = row_space_basis(gradients)
        residual = result.step - basis @ (basis.T @ result.step)

        self.assertEqual(result.branch, "feasible")
        self.assertLess(np.linalg.norm(residual), 1e-8)


if __name__ == "__main__":
    unittest.main()
