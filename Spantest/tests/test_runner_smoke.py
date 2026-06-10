from pathlib import Path
import contextlib
import io
import unittest

from Spantest.run_experiment import run_experiment


class RunnerSmokeTests(unittest.TestCase):
    def test_smoke_experiment_writes_curves_and_timing(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_experiment(
                output_root=Path(tmp_dir),
                run_name="smoke",
                solvers=("span", "active"),
                seed=0,
                T=5,
                grad_T=5,
                num_new_data=5,
                episode=2,
                update_time_per_episode=1,
                num_update_time=2,
                q_update_time=1,
                window=20,
                make_plots=True,
                verbose=False,
            )

            self.assertTrue(Path(run_dir, "summary.csv").exists())
            self.assertTrue(Path(run_dir, "timing.csv").exists())
            self.assertTrue(Path(run_dir, "curves.mat").exists())
            self.assertTrue(Path(run_dir, "objective_cost.png").exists())
            self.assertTrue(Path(run_dir, "objective_cost.pdf").exists())
            self.assertTrue(Path(run_dir, "solve_time.png").exists())
            self.assertTrue(Path(run_dir, "solve_time.pdf").exists())

    def test_verbose_experiment_prints_progress(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            stream = io.StringIO()
            with contextlib.redirect_stdout(stream):
                run_experiment(
                    output_root=Path(tmp_dir),
                    run_name="verbose_smoke",
                    solvers=("span",),
                    seed=0,
                    T=5,
                    grad_T=5,
                    num_new_data=5,
                    episode=2,
                    update_time_per_episode=1,
                    num_update_time=2,
                    q_update_time=1,
                    window=20,
                    make_plots=False,
                    verbose=True,
                )

            output = stream.getvalue()
            self.assertIn("[Spantest] output_dir=", output)
            self.assertIn("[Spantest] solver=span start", output)
            self.assertIn("[Spantest][span] episode=1/2 update=1/2", output)
            self.assertIn("decision_dim=", output)
            self.assertIn("solve_time_sec=", output)
            self.assertIn("[Spantest] done output_dir=", output)

    def test_dual_solver_runs_through_experiment(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_experiment(
                output_root=Path(tmp_dir),
                run_name="dual_smoke",
                solvers=("dual",),
                seed=0,
                T=5,
                grad_T=5,
                num_new_data=5,
                episode=2,
                update_time_per_episode=1,
                num_update_time=2,
                q_update_time=1,
                window=20,
                make_plots=False,
                verbose=False,
            )

            self.assertTrue(Path(run_dir, "summary.csv").exists())
            self.assertTrue(Path(run_dir, "timing.csv").exists())


if __name__ == "__main__":
    unittest.main()
