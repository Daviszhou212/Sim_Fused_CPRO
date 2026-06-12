import unittest
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from heuristic_mimo_baseline import (
    DEFAULT_CONFIG,
    MIMO_ACTION_DIM,
    MIMO_STATE_DIM,
    MIMO_UE_NUM,
    queue_aware_action,
    run_heuristic_rollout,
)


class QueueAwareActionTest(unittest.TestCase):
    def test_queue_aware_action_prioritizes_larger_delay(self):
        state = np.zeros(MIMO_STATE_DIM, dtype=np.float64)
        state[-MIMO_UE_NUM:] = np.array([0.0, 0.5, 2.0, 5.0], dtype=np.float64)

        action = queue_aware_action(state, DEFAULT_CONFIG)
        power = action[:MIMO_UE_NUM]

        self.assertEqual(action.shape, (MIMO_ACTION_DIM,))
        self.assertTrue(np.all(power > 0.0))
        self.assertLessEqual(float(np.max(power)), DEFAULT_CONFIG.max_power_per_user)
        self.assertLessEqual(float(np.sum(power)), DEFAULT_CONFIG.max_total_power)
        self.assertTrue(np.all(np.diff(power) >= -1e-12))
        self.assertEqual(float(action[-1]), DEFAULT_CONFIG.reg_factor)

    def test_short_rollout_returns_finite_episode_metrics(self):
        result = run_heuristic_rollout(
            episode_count=2,
            steps_per_episode=5,
            seed=0,
            config=DEFAULT_CONFIG,
            log_interval_episodes=1,
            verbose=False,
        )

        self.assertEqual(result.objective_cost.shape, (2,))
        self.assertEqual(result.avg_delay_per_user.shape, (2,))
        self.assertTrue(np.all(np.isfinite(result.objective_cost)))
        self.assertTrue(np.all(np.isfinite(result.avg_delay_per_user)))


if __name__ == "__main__":
    unittest.main()
