from pathlib import Path
import unittest

from Spantest.large_actor_dual_benchmark import run_benchmark


class LargeActorDualBenchmarkTests(unittest.TestCase):
    def test_benchmark_writes_summary_and_plot(self):
        import tempfile

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = run_benchmark(
                output_root=Path(tmp_dir),
                run_name="benchmark_smoke",
                theta_dims=(32, 64),
                repeats=1,
                span_cvxpy_max_dim=64,
            )

            self.assertTrue(Path(run_dir, "summary.csv").exists())
            self.assertTrue(Path(run_dir, "dual_time_vs_actor_dim.png").exists())
            self.assertTrue(Path(run_dir, "dual_time_vs_actor_dim.pdf").exists())


if __name__ == "__main__":
    unittest.main()
