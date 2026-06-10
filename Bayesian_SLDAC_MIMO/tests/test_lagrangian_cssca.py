import unittest

import numpy as np


class LagrangianCsscaTest(unittest.TestCase):
    def test_objective_only_descent_returns_finite_update(self):
        from Bayesian_SLDAC_MIMO.lagrangian_cssca import update_policy

        func_value = np.array([1.0], dtype=np.float64)
        grad = np.array([[2.0, 0.0]], dtype=np.float64)
        theta = np.zeros(2, dtype=np.float64)

        result = update_policy(func_value, grad, theta, tau_reward=1.0, tau_cost=1.0, return_info=True)

        self.assertTrue(result.success)
        self.assertTrue(np.isfinite(result.theta_bar).all())
        self.assertLess(result.theta_bar[0], 0.0)
        self.assertGreater(result.step_norm, 0.0)

    def test_failure_falls_back_to_current_theta(self):
        from Bayesian_SLDAC_MIMO.lagrangian_cssca import update_policy

        func_value = np.array([np.nan, 1.0], dtype=np.float64)
        grad = np.array([[1.0], [1.0]], dtype=np.float64)
        theta = np.array([3.0], dtype=np.float64)

        result = update_policy(func_value, grad, theta, tau_reward=1.0, tau_cost=1.0, return_info=True)

        self.assertFalse(result.success)
        self.assertEqual(result.status, "fallback_nonfinite_input")
        np.testing.assert_allclose(result.theta_bar, theta)


if __name__ == "__main__":
    unittest.main()
