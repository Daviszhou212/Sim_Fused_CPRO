import unittest

import numpy as np


class LegacySLDACBufferTest(unittest.TestCase):
    def test_fixed_two_t_training_window_and_average_window_shift(self):
        from MultiCell_MIMO.buffer import LegacySLDACBuffer

        buffer = LegacySLDACBuffer(
            t_horizon=2,
            num_new_data=2,
            state_dim=2,
            action_dim=1,
            cost_dim=3,
            window=3,
        )

        for idx in range(6):
            buffer.store_experiences(
                state=np.full(2, idx, dtype=np.float64),
                action=np.array([idx], dtype=np.float64),
                costs=np.full(3, idx, dtype=np.float64),
                next_state=np.full(2, idx + 0.5, dtype=np.float64),
                aver_objective=float(idx),
                aver_cost=float(idx + 10),
            )

        state, action, costs, next_state, aver_objective, aver_cost = buffer.take_experiences()

        np.testing.assert_allclose(state[:, 0], np.array([2, 3, 4, 5], dtype=np.float64))
        np.testing.assert_allclose(action[:, 0], np.array([2, 3, 4, 5], dtype=np.float64))
        np.testing.assert_allclose(costs[:, 0], np.array([2, 3, 4, 5], dtype=np.float64))
        np.testing.assert_allclose(next_state[:, 0], np.array([2.5, 3.5, 4.5, 5.5], dtype=np.float64))
        np.testing.assert_allclose(aver_objective, np.array([3, 4, 5], dtype=np.float64))
        np.testing.assert_allclose(aver_cost, np.array([13, 14, 15], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
