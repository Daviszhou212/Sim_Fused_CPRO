import unittest

import numpy as np


class CsscaTest(unittest.TestCase):
    def test_lagrangian_dual_objective_branch_returns_finite_theta(self):
        from MultiCell_MIMO.cssca import solve_cssca_update

        theta = np.zeros(2, dtype=np.float64)
        func_value = np.asarray([1.0, -0.5], dtype=np.float64)
        grad = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
        next_theta, info = solve_cssca_update(
            func_value,
            grad,
            theta,
            tau_objective=1.0,
            tau_constraint=1.0,
            solver="lagrangian_dual",
        )

        self.assertEqual(info["branch"], "objective")
        self.assertTrue(np.isfinite(next_theta).all())

    def test_lagrangian_dual_feasible_branch_returns_finite_theta(self):
        from MultiCell_MIMO.cssca import solve_cssca_update

        theta = np.zeros(2, dtype=np.float64)
        func_value = np.asarray([1.0, 2.0], dtype=np.float64)
        grad = np.asarray([[0.0, 0.0], [1.0, 0.0]], dtype=np.float64)
        next_theta, info = solve_cssca_update(
            func_value,
            grad,
            theta,
            tau_objective=1.0,
            tau_constraint=1.0,
            solver="lagrangian_dual",
        )

        self.assertEqual(info["branch"], "feasible")
        self.assertTrue(np.isfinite(next_theta).all())

    def test_invalid_shapes_fail_fast(self):
        from MultiCell_MIMO.cssca import solve_cssca_update

        with self.assertRaises(ValueError):
            solve_cssca_update(
                np.zeros(2),
                np.zeros((3, 2)),
                np.zeros(2),
                tau_objective=1.0,
                tau_constraint=1.0,
            )


if __name__ == "__main__":
    unittest.main()
