import tempfile
import unittest
from pathlib import Path

import torch


class CheckpointTest(unittest.TestCase):
    def test_checkpoint_disabled_writes_no_file(self):
        from MultiCell_MIMO.checkpoint import save_checkpoint
        from MultiCell_MIMO.config import build_default_config

        config = build_default_config()
        config["save_final_checkpoint"] = 0
        with tempfile.TemporaryDirectory() as tmpdir:
            result = save_checkpoint(
                checkpoint_root=tmpdir,
                config=config,
                state_dict={"weight": torch.ones(1)},
                stats={"objective_history": [1.0], "cost_history": [0.5]},
                episode_index=1,
                reason="final",
            )
            self.assertIsNone(result)
            self.assertEqual(list(Path(tmpdir).glob("*.pt")), [])

    def test_checkpoint_schema_records_runtime_contract(self):
        from MultiCell_MIMO.checkpoint import save_checkpoint
        from MultiCell_MIMO.config import build_default_config

        config = build_default_config()
        config["save_final_checkpoint"] = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            path = save_checkpoint(
                checkpoint_root=tmpdir,
                config=config,
                state_dict={"weight": torch.ones(1)},
                stats={"objective_history": [1.0], "cost_history": [0.5]},
                episode_index=1,
                reason="final",
            )
            payload = torch.load(path, map_location="cpu")

        self.assertEqual(payload["schema_version"], 1)
        self.assertEqual(payload["config"]["critic_backend"], "centralized")
        self.assertEqual(payload["config"]["critic_target_mode"], "source_compatible")
        self.assertEqual(payload["config"]["actor_parameterization"], "shared")
        self.assertEqual(payload["config"]["log_std_mode"], "shared_cell")
        self.assertIn("state_dict", payload)

    def test_checkpoint_refuses_duplicate_filename_by_default(self):
        from MultiCell_MIMO.checkpoint import save_checkpoint
        from MultiCell_MIMO.config import build_default_config

        config = build_default_config()
        config["save_final_checkpoint"] = 1
        with tempfile.TemporaryDirectory() as tmpdir:
            save_checkpoint(
                checkpoint_root=tmpdir,
                config=config,
                state_dict={"weight": torch.ones(1)},
                stats={"objective_history": [1.0], "cost_history": [0.5]},
                episode_index=1,
                reason="final",
            )
            with self.assertRaises(FileExistsError):
                save_checkpoint(
                    checkpoint_root=tmpdir,
                    config=config,
                    state_dict={"weight": torch.ones(1)},
                    stats={"objective_history": [1.0], "cost_history": [0.5]},
                    episode_index=1,
                    reason="final",
                )


if __name__ == "__main__":
    unittest.main()
