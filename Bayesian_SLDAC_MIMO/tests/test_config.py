import os
import unittest


class ConfigTest(unittest.TestCase):
    def test_defaults_match_sldac_code_mimo1(self):
        from Bayesian_SLDAC_MIMO.config import SLDAC_RUNS, make_run_config

        cfg = make_run_config("b100_q1")

        self.assertEqual(cfg.example_name, "MIMO")
        self.assertEqual(cfg.seed, 0)
        self.assertEqual(cfg.device, "cpu")
        self.assertEqual(cfg.alpha_pow, 0.6)
        self.assertEqual(cfg.beta_pow, 0.7)
        self.assertEqual(cfg.eta_pow, 0.01)
        self.assertEqual(cfg.gamma_pow_reward, 0.3)
        self.assertEqual(cfg.gamma_pow_cost, 0.3)
        self.assertEqual(cfg.tau_reward, 1.0)
        self.assertEqual(cfg.tau_cost, 1.0)
        self.assertEqual(cfg.T, 500)
        self.assertEqual(cfg.grad_T, 500)
        self.assertEqual(cfg.num_new_data, 100)
        self.assertEqual(cfg.window, 10000)
        self.assertEqual(cfg.episode, 60)
        self.assertEqual(cfg.update_time_per_episode, 10)
        self.assertEqual(cfg.num_update_time, 600)
        self.assertEqual(cfg.Q_update_time, 1)
        self.assertEqual(cfg.MAX_STEPS, 61000)
        self.assertEqual(cfg.ensemble_size, 5)
        self.assertEqual(cfg.ensemble_init_mode, "shared")
        self.assertEqual(cfg.bootstrap_mask_prob, 1.0)
        self.assertEqual(cfg.beta_uncertainty, 0.0)
        self.assertEqual(cfg.critic_lr_base, 0.1)
        self.assertEqual(SLDAC_RUNS["b100_q1"], ("Bayesian SLDAC, batchsize=100, q=1", 500, 500, 100, 1))
        self.assertEqual(SLDAC_RUNS["b100_q10"], ("Bayesian SLDAC, batchsize=100, q=10", 500, 500, 100, 10))
        self.assertEqual(SLDAC_RUNS["b100_q5"], ("Bayesian SLDAC, batchsize=100, q=5", 500, 500, 100, 5))
        self.assertEqual(SLDAC_RUNS["b500_q10"], ("Bayesian SLDAC, source b500_q10 setting", 50, 100, 100, 10))

    def test_episode_override_recomputes_max_steps(self):
        from Bayesian_SLDAC_MIMO.config import make_run_config

        cfg = make_run_config("b100_q1", episode_override=100)

        self.assertEqual(cfg.episode, 100)
        self.assertEqual(cfg.num_update_time, 1000)
        self.assertEqual(cfg.MAX_STEPS, 101000)

    def test_artifact_paths_are_isolated(self):
        from Bayesian_SLDAC_MIMO.artifact_paths import PACKAGE_ROOT, make_run_paths

        test_root = os.path.join(PACKAGE_ROOT, "Trash", "path_test")
        paths = make_run_paths(
            "compare_b100_q1_100ep_unit",
            output_root=os.path.join(test_root, "outputs"),
            log_root=os.path.join(test_root, "logs"),
            trash_root=os.path.join(test_root, "trash"),
        )
        package_root = os.path.abspath(PACKAGE_ROOT)
        repo_root = os.path.dirname(package_root)
        legacy_mimo_outputs = os.path.abspath(os.path.join(repo_root, "Squash", "MIMO", "outputs"))
        legacy_checkpoints = os.path.abspath(os.path.join(repo_root, "checkpoints", "SLDAC"))
        legacy_sldac_code = os.path.abspath(os.path.join(repo_root, "SLDAC_code", "MIMO1"))

        for path in (paths.output_dir, paths.log_dir, paths.trash_dir):
            resolved = os.path.abspath(path)
            self.assertTrue(resolved.startswith(package_root + os.sep) or resolved == package_root)
            self.assertFalse(resolved.startswith(legacy_sldac_code + os.sep))
            self.assertFalse(resolved.startswith(legacy_mimo_outputs + os.sep))
            self.assertFalse(resolved.startswith(legacy_checkpoints + os.sep))


if __name__ == "__main__":
    unittest.main()
