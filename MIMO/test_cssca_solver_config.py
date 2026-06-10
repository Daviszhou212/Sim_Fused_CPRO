import importlib
import os
import sys
import unittest

import numpy as np


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)


CSSCA_ENTRY_MODULES = (
    "run_mimo_sldac",
    "run_mimo_fused_cpro",
    "run_mimo_fused_cpro_cosrho",
    "run_mimo_fused_cpro_rho_new",
    "run_mimo_hrl",
    "run_mimo_prcrl",
)


class CsscaSolverConfigTest(unittest.TestCase):
    def test_entry_defaults_expose_cvx_cssca_solver(self):
        for module_name in CSSCA_ENTRY_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                config = module.build_python_config()
                self.assertEqual(config["cssca_solver"], "cvx")
                self.assertIn("cssca_solver", module.PROTECTED_CLI_FIELDS)

    def test_dual_solver_objective_branch_uses_closed_form_unconstrained_step(self):
        from cssca_dual_solver import solve_dual_cssca_update

        func_value = np.array([0.0, -100.0], dtype=np.float64)
        grad = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        theta = np.zeros(2, dtype=np.float64)

        theta_bar, info = solve_dual_cssca_update(
            func_value,
            grad,
            theta,
            tau_reward=1.0,
            tau_cost=1.0,
        )

        self.assertEqual(info["branch"], "objective")
        self.assertTrue(np.allclose(theta_bar, np.array([-1.0, 0.0]), atol=1e-8))

    def test_dual_solver_feasible_branch_minimizes_constraint_violation(self):
        from cssca_dual_solver import solve_dual_cssca_update

        func_value = np.array([0.0, 4.0], dtype=np.float64)
        grad = np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float64)
        theta = np.zeros(2, dtype=np.float64)

        theta_bar, info = solve_dual_cssca_update(
            func_value,
            grad,
            theta,
            tau_reward=1.0,
            tau_cost=1.0,
        )

        self.assertEqual(info["branch"], "feasible")
        self.assertTrue(np.allclose(theta_bar, np.array([-1.0, 0.0]), atol=1e-8))
        self.assertAlmostEqual(info["feasible_x"], 3.0, places=8)

    def test_fused_policy_update_uses_dual_solver_without_simplex(self):
        from Fused_CPRO import _policy_update

        func_value = np.array([0.0, -100.0], dtype=np.float64)
        grad = np.array([[2.0, 0.0], [0.0, 0.0]], dtype=np.float64)
        theta = np.zeros(2, dtype=np.float64)

        theta_bar = _policy_update(
            func_value,
            grad,
            theta,
            tau_reward=1.0,
            tau_cost=1.0,
            cssca_solver="dual",
        )

        self.assertTrue(np.allclose(theta_bar, np.array([-1.0, 0.0]), atol=1e-8))


if __name__ == "__main__":
    unittest.main()
