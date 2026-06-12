import unittest

import numpy as np


class MultiCellDbActionSemanticsTest(unittest.TestCase):
    def test_snr_db_action_executes_same_power_semantics_as_legacy_power_action(self):
        from MultiCell_MIMO.environment import ACTION_EPS, MultiCellMIMOEnv
        from MultiCell_MIMO.sldac import (
            multicell_buffer_action_from_info,
            multicell_power_action_to_db_action,
        )

        legacy_env = MultiCellMIMOEnv(seed=13, nt=2, cell_count=2, users_per_cell=2)
        db_env = MultiCellMIMOEnv(
            seed=13,
            nt=2,
            cell_count=2,
            users_per_cell=2,
            action_interface="snr_db",
        )
        legacy_env.reset()
        db_env.reset()

        raw_power_action = np.array(
            [
                -1.0,
                legacy_env.noise_power / 10.0,
                -2.0,
                0.5,
                99.0,
                0.2,
            ],
            dtype=np.float64,
        )
        db_action = multicell_power_action_to_db_action(raw_power_action, db_env)

        legacy_next, legacy_objective, _, legacy_info = legacy_env.step(raw_power_action.copy())
        db_next, db_objective, _, db_info = db_env.step(db_action)
        buffer_action = multicell_buffer_action_from_info(raw_power_action, db_info, "snr_db")

        expected_executed_power = np.array(
            [
                ACTION_EPS,
                ACTION_EPS,
                ACTION_EPS,
                0.5,
                99.0,
                0.2,
            ],
            dtype=np.float64,
        )

        self.assertAlmostEqual(legacy_objective, db_objective)
        np.testing.assert_allclose(legacy_next, db_next, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(legacy_info["cell_cost"], db_info["cell_cost"], rtol=0.0, atol=0.0)
        np.testing.assert_allclose(db_info["executed_power_action"], expected_executed_power, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(buffer_action, expected_executed_power, rtol=1e-12, atol=1e-12)

    def test_legacy_buffer_action_uses_executed_action_to_match_sldac_code(self):
        from MultiCell_MIMO.sldac import multicell_buffer_action_from_info

        action = np.array([-1.0, 0.5, 0.2], dtype=np.float64)
        info = {"executed_power_action": np.array([1e-6, 0.5, 0.2], dtype=np.float64)}

        buffer_action = multicell_buffer_action_from_info(action, info, "legacy_power")

        np.testing.assert_array_equal(buffer_action, info["executed_power_action"])


if __name__ == "__main__":
    unittest.main()
