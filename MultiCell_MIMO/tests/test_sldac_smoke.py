import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch


class SldacSmokeTest(unittest.TestCase):
    def _run_tiny_sldac(self, overrides):
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
                    "run_id": "unit_{0}".format(overrides.get("critic_backend", "centralized")),
                }
            )
            config.update(overrides)
            result = run_sldac(config)

            self.assertEqual(len(result["objective_history"]), 1)
            self.assertEqual(len(result["cost_history"]), 1)
            self.assertEqual(result["critic_backend"], config["critic_backend"])
            self.assertEqual(result["run_id"], config["run_id"])
            self.assertTrue(np.isfinite(result["objective_history"]).all())
            self.assertTrue(np.isfinite(result["cost_history"]).all())
            self.assertEqual(list(Path(checkpoint_root).rglob("*.pt")), [])

    def test_sldac_smoke_runs_with_centralized_critic(self):
        self._run_tiny_sldac({})

    def test_sldac_smoke_runs_with_tree_critic(self):
        self._run_tiny_sldac({"critic_backend": "tree", "critic_target_mode": "tex_strict"})

    def test_tree_critic_checkpoint_records_tree_modules(self):
        from MultiCell_MIMO.config import build_default_config
        from MultiCell_MIMO.sldac import run_sldac

        with tempfile.TemporaryDirectory() as output_root, tempfile.TemporaryDirectory() as checkpoint_root:
            config = build_default_config()
            config.update(
                {
                    "seed": 5,
                    "device": "cpu",
                    "nt": 2,
                    "cell_count": 2,
                    "users_per_cell": 1,
                    "episode": 1,
                    "update_time_per_episode": 1,
                    "t_horizon": 3,
                    "grad_batch_size": 3,
                    "num_new_data": 2,
                    "window": 20,
                    "hidden_dims": (8,),
                    "critic_hidden_dims": (8,),
                    "critic_backend": "tree",
                    "critic_target_mode": "tex_strict",
                    "save_final_checkpoint": 1,
                    "output_root": output_root,
                    "checkpoint_root": checkpoint_root,
                    "run_id": "tree_checkpoint",
                }
            )
            run_sldac(config)
            checkpoint_paths = list(Path(checkpoint_root).rglob("*.pt"))
            self.assertEqual(len(checkpoint_paths), 1)
            payload = torch.load(checkpoint_paths[0], map_location="cpu")
        keys = set(payload["state_dict"].keys())
        self.assertEqual(payload["config"]["critic_backend"], "tree")
        self.assertIn("critic.encoder.0.weight", keys)
        self.assertIn("critic.target_encoder.0.weight", keys)
        self.assertTrue(any(key.startswith("critic.heads.") for key in keys))
        self.assertTrue(any(key.startswith("critic.target_heads.") for key in keys))


if __name__ == "__main__":
    unittest.main()
