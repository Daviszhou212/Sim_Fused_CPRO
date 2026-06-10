from pathlib import Path
import subprocess
import sys
import unittest

from Spantest.main import (
    DEFAULT_SOLVERS,
    PRINT_INTERVAL,
    RUN_LAGRANGIAN_SLDAC,
    RUN_ORIGINAL_SLDAC,
    SLDAC_MIMO1_CONFIG,
    run_spantest_main,
)


class MainEntryTests(unittest.TestCase):
    def test_default_entry_config_matches_sldac_code_mimo1_defaults(self):
        self.assertEqual(SLDAC_MIMO1_CONFIG["seed"], 0)
        self.assertEqual(SLDAC_MIMO1_CONFIG["T"], 500)
        self.assertEqual(SLDAC_MIMO1_CONFIG["grad_T"], 500)
        self.assertEqual(SLDAC_MIMO1_CONFIG["num_new_data"], 100)
        self.assertEqual(SLDAC_MIMO1_CONFIG["window"], 10000)
        self.assertEqual(SLDAC_MIMO1_CONFIG["episode"], 60)
        self.assertEqual(SLDAC_MIMO1_CONFIG["update_time_per_episode"], 10)
        self.assertEqual(SLDAC_MIMO1_CONFIG["num_update_time"], 600)
        self.assertEqual(SLDAC_MIMO1_CONFIG["q_update_time"], 1)
        self.assertGreaterEqual(PRINT_INTERVAL, 1)
        self.assertEqual(SLDAC_MIMO1_CONFIG["print_interval"], PRINT_INTERVAL)

    def test_default_solvers_skip_original_sldac_when_switch_is_off(self):
        self.assertFalse(RUN_ORIGINAL_SLDAC)
        self.assertNotIn("full", DEFAULT_SOLVERS)
        self.assertGreaterEqual(len(DEFAULT_SOLVERS), 1)

    def test_default_solvers_include_lagrangian_when_switch_is_on(self):
        self.assertTrue(RUN_LAGRANGIAN_SLDAC)
        self.assertIn("dual", DEFAULT_SOLVERS)

    def test_main_entry_runs_a_complete_smoke_pipeline(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_spantest_main(
                output_root=Path(tmp_dir),
                run_name="entry_smoke",
                use_smoke_config=True,
                solvers=("span", "active"),
                make_plots=False,
                verbose=False,
            )

            self.assertTrue(Path(run_dir, "summary.csv").exists())
            self.assertTrue(Path(run_dir, "timing.csv").exists())
            self.assertTrue(Path(run_dir, "curves.mat").exists())
            self.assertTrue(Path(run_dir, "metadata.json").exists())

    def test_main_py_can_be_run_directly(self):
        import tempfile

        script_path = Path(__file__).resolve().parents[1] / "main.py"
        with tempfile.TemporaryDirectory() as tmp_dir:
            completed = subprocess.run(
                [
                    sys.executable,
                    str(script_path),
                    "--smoke",
                    "--output-root",
                    tmp_dir,
                    "--run-name",
                    "direct_script_smoke",
                    "--solvers",
                    "span",
                    "--no-plots",
                ],
                cwd=Path(__file__).resolve().parents[2],
                text=True,
                capture_output=True,
                timeout=60,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertIn("[Spantest] loading dependencies", completed.stdout)
            self.assertTrue(Path(tmp_dir, "direct_script_smoke", "summary.csv").exists())


if __name__ == "__main__":
    unittest.main()
