import unittest

import numpy as np

import SCAOPO
import SLDAC
from environment import Environment_MIMO, LEGACY_ACTION_MIN


class DbActionBufferSemanticsTest(unittest.TestCase):
    def test_mimo_buffer_action_matches_executed_power_action(self):
        env = Environment_MIMO(seed=0, Nt=2, UE_num=4)
        raw_action = np.array([-1.0, env.noise_power / 10.0, 0.5, 99.0, -2.0])

        env_action = SLDAC.mimo_power_action_to_db_action(raw_action, env)
        _, reward, _, info = env.step(env_action)

        expected = np.array(
            [
                LEGACY_ACTION_MIN,
                env.noise_power / 10.0,
                0.5,
                99.0,
                LEGACY_ACTION_MIN,
            ]
        )

        for module in (SLDAC, SCAOPO):
            with self.subTest(module=module.__name__):
                buffer_action = module.mimo_buffer_action_from_info(raw_action, info, "MIMO")
                np.testing.assert_allclose(buffer_action, expected, rtol=1e-12, atol=1e-12)

        self.assertAlmostEqual(reward, np.sum(expected[: env.UE_num]))
        np.testing.assert_allclose(info["executed_power_action"], expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
