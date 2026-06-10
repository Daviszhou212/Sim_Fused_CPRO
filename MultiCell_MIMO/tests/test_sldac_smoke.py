import tempfile
import unittest
from pathlib import Path

import numpy as np


class SldacSmokeTest(unittest.TestCase):
    def test_sldac_smoke_runs_with_temp_roots_and_no_checkpoint(self):
        from MultiCell_MIMO.config import build_default_config
        from MultiCell_MIMO.sldac import run_sldac

        with tempfile.TemporaryDirectory() as output_root, tempfile.TemporaryDirectory() as checkpoint_root:
            config = build_default_config()
            config.update(
                {
                    "seed": 3,
                    "device": "cpu",
                    "nt": 2,
                    "cell_count": 2,
                    "users_per_cell": 1,
                    "episode": 1,
                    "update_time_per_episode": 1,
                    "t_horizon": 3,
                    "grad_batch_size": 3,
                    "num_new_data": 2,
                    "q_update_time": 1,
                    "window": 20,
                    "hidden_dims": (8,),
                    "critic_hidden_dims": (8,),
                    "save_final_checkpoint": 0,
                    "output_root": output_root,
                    "checkpoint_root": checkpoint_root,
                }
            )
            result = run_sldac(config)

            self.assertEqual(len(result["objective_history"]), 1)
            self.assertEqual(len(result["cost_history"]), 1)
            self.assertTrue(np.isfinite(result["objective_history"]).all())
            self.assertTrue(np.isfinite(result["cost_history"]).all())
            self.assertEqual(list(Path(checkpoint_root).rglob("*.pt")), [])


if __name__ == "__main__":
    unittest.main()
