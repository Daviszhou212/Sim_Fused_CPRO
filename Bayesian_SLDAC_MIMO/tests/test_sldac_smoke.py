import os
import unittest
from unittest.mock import patch


class SldacSmokeTest(unittest.TestCase):
    def test_small_compare_run_writes_only_to_trash(self):
        from Bayesian_SLDAC_MIMO.artifact_paths import PACKAGE_ROOT
        from Bayesian_SLDAC_MIMO.config import make_run_config
        from Bayesian_SLDAC_MIMO.sldac import run_compare

        cfg = make_run_config(
            "b100_q1",
            episode_override=1,
            overrides={
                "T": 3,
                "grad_T": 3,
                "num_new_data": 2,
                "window": 10,
                "Q_update_time": 1,
                "update_time_per_episode": 1,
                "ensemble_size": 2,
                "beta_uncertainty": 0.25,
            },
        )
        result = run_compare(cfg, output_root=os.path.join(PACKAGE_ROOT, "Trash", "unit_smoke"))

        self.assertTrue(os.path.abspath(result.output_dir).startswith(os.path.abspath(os.path.join(PACKAGE_ROOT, "Trash"))))
        self.assertTrue(os.path.exists(os.path.join(result.output_dir, "legacy_reward_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(result.output_dir, "bayesian_reward_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(result.output_dir, "legacy_worst_constraint_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(result.output_dir, "bayesian_worst_constraint_b100_q1.mat")))
        self.assertIn("legacy", result.summary)
        self.assertIn("bayesian", result.summary)
        self.assertIn("objective_avg_final", result.summary["bayesian"])
        self.assertIn("average_cost_final", result.summary["bayesian"])
        self.assertIn("average_cost_violation_final", result.summary["bayesian"])
        self.assertIn("worst_user_constraint_residual_final", result.summary["bayesian"])
        self.assertIn("constraint_violation_final", result.summary["bayesian"])
        self.assertIn("ensemble_size", result.summary["bayesian"])
        self.assertIn("func_value", result.summary["diagnostics"]["bayesian"])
        self.assertIn("grad_norm_objective", result.summary["diagnostics"]["bayesian"])
        self.assertIn("grad_norm_constraints", result.summary["diagnostics"]["bayesian"])
        self.assertIn("q_used_batch_std", result.summary["diagnostics"]["bayesian"])
        self.assertIn("q_saturation_fraction", result.summary["diagnostics"]["bayesian"])

    def test_compare_default_output_uses_isolated_run_paths(self):
        from Bayesian_SLDAC_MIMO.artifact_paths import PACKAGE_ROOT, RunPaths
        from Bayesian_SLDAC_MIMO.config import make_run_config
        from Bayesian_SLDAC_MIMO.sldac import SldacRunResult, run_compare

        cfg = make_run_config("b100_q1", episode_override=1)
        output_dir = os.path.join(PACKAGE_ROOT, "Trash", "default_output_unit")
        fake_paths = RunPaths(
            run_id="default_output_unit",
            output_dir=output_dir,
            log_dir=os.path.join(PACKAGE_ROOT, "Trash", "default_output_unit_logs"),
            trash_dir=os.path.join(PACKAGE_ROOT, "Trash", "default_output_unit_trash"),
        )
        fake_result = SldacRunResult(
            reward_average_save=[1.0],
            cost_average_save=[1.2],
            diagnostics={
                "worst_user_constraint_residual": [0.0],
                "ensemble_size": 1,
                "beta_uncertainty": 0.0,
            },
        )

        with patch("Bayesian_SLDAC_MIMO.sldac.make_run_paths", return_value=fake_paths) as make_paths:
            with patch("Bayesian_SLDAC_MIMO.sldac.SLDAC_main", return_value=fake_result):
                result = run_compare(cfg)

        make_paths.assert_called_once()
        self.assertEqual(os.path.abspath(result.output_dir), os.path.abspath(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "legacy_reward_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "legacy_worst_constraint_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "summary.json")))

    def test_bayesian_only_run_does_not_write_legacy_outputs(self):
        from Bayesian_SLDAC_MIMO.artifact_paths import PACKAGE_ROOT
        from Bayesian_SLDAC_MIMO.config import make_run_config
        from Bayesian_SLDAC_MIMO.sldac import run_bayesian

        cfg = make_run_config(
            "b100_q1",
            episode_override=1,
            overrides={
                "T": 3,
                "grad_T": 3,
                "num_new_data": 2,
                "window": 10,
                "Q_update_time": 1,
                "update_time_per_episode": 1,
                "ensemble_size": 2,
            },
        )
        output_dir = os.path.join(PACKAGE_ROOT, "Trash", "bayesian_only_unit")
        result = run_bayesian(cfg, output_root=output_dir)

        self.assertEqual(os.path.abspath(result.output_dir), os.path.abspath(output_dir))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "bayesian_reward_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "bayesian_cost_b100_q1.mat")))
        self.assertTrue(os.path.exists(os.path.join(output_dir, "bayesian_worst_constraint_b100_q1.mat")))
        self.assertFalse(os.path.exists(os.path.join(output_dir, "legacy_reward_b100_q1.mat")))
        self.assertIn("bayesian", result.summary)
        self.assertNotIn("legacy", result.summary)


if __name__ == "__main__":
    unittest.main()
