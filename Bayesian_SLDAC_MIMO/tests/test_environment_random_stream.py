import unittest

import numpy as np


class EnvironmentRandomStreamTest(unittest.TestCase):
    def test_reset_and_step_are_deterministic_for_seed_zero(self):
        from Bayesian_SLDAC_MIMO.environment import Environment_MIMO

        env_a = Environment_MIMO(seed=0, Nt=8, UE_num=4)
        env_b = Environment_MIMO(seed=0, Nt=8, UE_num=4)

        state_a = env_a.reset()
        state_b = env_b.reset()
        np.testing.assert_allclose(state_a, state_b)

        action = np.array([0.5, 0.7, 1.0, 1.2, 0.25], dtype=np.float64)
        next_a, reward_a, done_a, info_a = env_a.step(action.copy())
        next_b, reward_b, done_b, info_b = env_b.step(action.copy())

        np.testing.assert_allclose(next_a, next_b)
        self.assertEqual(reward_a, reward_b)
        self.assertEqual(done_a, done_b)
        self.assertEqual(info_a.keys(), info_b.keys())
        for key in info_a:
            self.assertAlmostEqual(float(info_a[key]), float(info_b[key]))

    def test_action_lower_projection_and_no_upper_clip(self):
        from Bayesian_SLDAC_MIMO.environment import Environment_MIMO, project_mimo_action

        action = np.array([-2.0, 0.0, 3.5, 4.25, -0.1], dtype=np.float64)
        projected = project_mimo_action(action)

        np.testing.assert_allclose(projected, np.array([1e-6, 1e-6, 3.5, 4.25, 1e-6]))

        env = Environment_MIMO(seed=0, Nt=8, UE_num=4)
        env.reset()
        _, reward, _, _ = env.step(action)
        self.assertAlmostEqual(reward, 1e-6 + 1e-6 + 3.5 + 4.25)


if __name__ == "__main__":
    unittest.main()
