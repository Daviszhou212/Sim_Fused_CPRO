import unittest

import numpy as np
import torch


class LegacyActionModelTest(unittest.TestCase):
    def test_actor_mean_is_bounded_but_distribution_is_plain_gaussian(self):
        from Bayesian_SLDAC_MIMO.model import GaussianPolicy_MIMO

        torch.manual_seed(0)
        actor = GaussianPolicy_MIMO(state_dim=68, action_dim=5, device="cpu", num_new_data=4)
        states = torch.zeros((4, 68), dtype=torch.float32)
        actions = torch.zeros((4, 5), dtype=torch.float32)

        mu = actor.net(states)
        self.assertTrue(torch.all(mu > 0.0).item())
        self.assertTrue(torch.all(mu < 2.5).item())

        log_prob = actor.evaluate_action(states, actions)
        self.assertEqual(tuple(log_prob.shape), (4,))
        self.assertTrue(torch.isfinite(log_prob).all().item())
        self.assertEqual(getattr(actor, "actor_distribution"), "legacy_bounded_mean_plain_gaussian")

    def test_sampling_can_leave_mean_bounds(self):
        from Bayesian_SLDAC_MIMO.model import GaussianPolicy_MIMO

        torch.manual_seed(4)
        np.random.seed(4)
        actor = GaussianPolicy_MIMO(state_dim=68, action_dim=5, device="cpu", num_new_data=1)
        actor.log_std = torch.ones(5, dtype=torch.float32) * 1.5
        state = np.zeros((68,), dtype=np.float32)

        samples = np.asarray([actor.sample_action(state) for _ in range(128)])

        self.assertTrue(np.any(samples < 0.0) or np.any(samples > 2.5))

    def test_mimo_critic_output_scale_matches_sldac_code(self):
        from Bayesian_SLDAC_MIMO.model import CriticNetMIMO

        critic = CriticNetMIMO(state_dim=2, action_dim=1, device="cpu")
        for param in critic.parameters():
            param.data.zero_()
        critic.fc3.bias.data.fill_(0.5)
        state = torch.zeros((1, 2), dtype=torch.float32)
        action = torch.zeros((1, 1), dtype=torch.float32)

        value = critic(state, action)

        expected = 10.0 * torch.tanh(0.001 * torch.tensor([[0.5]], dtype=torch.float32))
        torch.testing.assert_close(value, expected)


if __name__ == "__main__":
    unittest.main()
