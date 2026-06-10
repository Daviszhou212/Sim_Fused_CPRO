import unittest

import numpy as np


class EnvironmentTest(unittest.TestCase):
    def test_reset_step_shapes_and_costs_are_finite(self):
        from MultiCell_MIMO.environment import MultiCellMIMOEnv

        env = MultiCellMIMOEnv(seed=7, nt=4, cell_count=3, users_per_cell=2)
        state = env.reset()

        self.assertEqual(state.shape, (env.state_dim,))
        self.assertEqual(env.action_dim, 9)
        self.assertEqual(env.constraint_dim, 6)
        self.assertEqual(env.local_actor_state_dim, 2 * env.users_per_cell * env.nt + env.users_per_cell)
        self.assertEqual(env.local_critic_state_dim, 2 * env.users_per_cell * env.cell_count * env.nt + env.users_per_cell)

        action = np.full(env.action_dim, 0.5, dtype=np.float64)
        next_state, objective_cost, done, info = env.step(action)

        self.assertEqual(next_state.shape, (env.state_dim,))
        self.assertFalse(done)
        self.assertTrue(np.isfinite(objective_cost))
        self.assertTrue(np.isfinite(info["cost"]))
        self.assertIn("cost_6", info)
        self.assertEqual(env.local_actor_observations().shape, (env.cell_count, env.local_actor_state_dim))
        self.assertEqual(env.local_critic_observations().shape, (env.cell_count, env.local_critic_state_dim))
        self.assertEqual(
            env.batch_local_critic_observations(np.stack((state, next_state), axis=0)).shape,
            (2, env.cell_count, env.local_critic_state_dim),
        )

    def test_invalid_action_dimension_raises(self):
        from MultiCell_MIMO.environment import MultiCellMIMOEnv

        env = MultiCellMIMOEnv(seed=0, nt=2, cell_count=2, users_per_cell=1)
        env.reset()
        with self.assertRaises(ValueError):
            env.step(np.ones(env.action_dim + 1, dtype=np.float64))

    def test_legacy_seed_step_stream_resets_and_advances(self):
        from MultiCell_MIMO.environment import MultiCellMIMOEnv

        env = MultiCellMIMOEnv(seed=11, nt=2, cell_count=2, users_per_cell=1)
        env.reset()
        self.assertEqual(env.seed_step, 11)

        action = np.full(env.action_dim, 0.5, dtype=np.float64)
        env.step(action)
        self.assertEqual(env.seed_step, 12)

        state_after_reset = env.reset().copy()
        self.assertEqual(env.seed_step, 11)

        env2 = MultiCellMIMOEnv(seed=11, nt=2, cell_count=2, users_per_cell=1)
        np.testing.assert_allclose(state_after_reset, env2.reset())


if __name__ == "__main__":
    unittest.main()
