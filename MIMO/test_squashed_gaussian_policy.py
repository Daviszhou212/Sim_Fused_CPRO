import unittest

import numpy as np
import torch

from Fused_CPRO import (
    HeuristicGaussianPolicy,
    _build_dk_policy,
    _build_mixture_log_prob,
    _log_prob_batch,
)
from model import (
    ACTION_EPS,
    LEGACY_ACTOR_DISTRIBUTION,
    MIMO_POWER_MAX,
    SQUASHED_ACTOR_DISTRIBUTION,
    GaussianPolicy_MIMO,
    mimo_inverse_action_and_log_det,
    mimo_transform_raw_action,
)


class SquashedGaussianMimoTest(unittest.TestCase):
    def test_mimo_actor_bounds_and_manual_log_prob(self):
        actor = GaussianPolicy_MIMO(state_dim=7, action_dim=5, device="cpu", num_new_data=4)
        state_torch = torch.randn(6, 7, dtype=torch.float)
        action_torch = actor.sample_action_tensor(
            state_torch,
            reparameterized=True,
            use_mean=False,
            track_log_std_grad=True,
        )

        self.assertTrue(torch.all(action_torch[:, :-1] > ACTION_EPS).item())
        self.assertTrue(torch.all(action_torch[:, :-1] < MIMO_POWER_MAX).item())
        self.assertTrue(torch.all(action_torch[:, -1] > ACTION_EPS).item())
        self.assertTrue(torch.isfinite(actor.evaluate_action(state_torch, action_torch)).all().item())

        raw_loc = actor.net(state_torch)
        raw_action, log_det, valid = mimo_inverse_action_and_log_det(action_torch)
        expected = torch.distributions.normal.Normal(
            raw_loc,
            torch.exp(actor.log_std).view(1, -1).expand_as(raw_loc),
        ).log_prob(raw_action).sum(dim=1) - log_det
        actual = actor.evaluate_action_with_log_std(state_torch, action_torch, actor.log_std)

        self.assertTrue(valid.all().item())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

        boundary_action = action_torch.detach().clone()
        boundary_action[:, 0] = ACTION_EPS
        self.assertTrue(torch.isneginf(actor.evaluate_action(state_torch, boundary_action)).all().item())

        raw_probe = torch.zeros((1, 5), dtype=torch.float)
        raw_probe[:, -1] = 5.0
        transformed_probe = mimo_transform_raw_action(raw_probe)
        self.assertGreater(float(transformed_probe[0, -1].item()), MIMO_POWER_MAX)

    def test_fused_mixture_keeps_logsumexp_and_log_std_grad(self):
        actor = GaussianPolicy_MIMO(state_dim=7, action_dim=5, device="cpu", num_new_data=4)
        state_torch = torch.randn(5, 7, dtype=torch.float)
        action_torch = actor.sample_action_tensor(
            state_torch,
            reparameterized=False,
            use_mean=False,
            track_log_std_grad=False,
        ).detach()

        boundary_mean = np.asarray([MIMO_POWER_MAX, MIMO_POWER_MAX, MIMO_POWER_MAX, MIMO_POWER_MAX, 0.25], dtype=np.float64)
        dk_policy = HeuristicGaussianPolicy(lambda state: boundary_mean, 5, "cpu", transform_kind="mimo")
        rho = np.asarray([0.7, 0.3], dtype=np.float64)
        lower_bounds = np.asarray([0.2, 1e-4], dtype=np.float64)

        log_mix, rho_torch, log_std_leaf = _build_mixture_log_prob(
            state_torch,
            action_torch,
            actor,
            [dk_policy],
            rho,
            lower_bounds,
        )
        log_prob_new, _ = _log_prob_batch(actor, state_torch, action_torch, require_grad=False)
        log_prob_dk = dk_policy.log_prob_batch(state_torch, action_torch)
        expected = torch.logsumexp(
            torch.stack(
                (
                    log_prob_new + torch.log(torch.tensor(rho[0], dtype=torch.float)),
                    log_prob_dk + torch.log(torch.tensor(rho[1], dtype=torch.float)),
                ),
                dim=1,
            ),
            dim=1,
        )
        self.assertTrue(torch.allclose(log_mix, expected, atol=1e-5, rtol=1e-5))

        (-log_mix.mean()).backward()
        self.assertIsNotNone(rho_torch.grad)
        self.assertTrue(torch.isfinite(rho_torch.grad).all().item())
        self.assertIsNotNone(log_std_leaf.grad)
        self.assertTrue(torch.isfinite(log_std_leaf.grad).all().item())

        sampled = dk_policy.sample_action(np.zeros((7,), dtype=np.float64))
        np.testing.assert_allclose(sampled, boundary_mean)
        boundary_batch = torch.tensor(np.tile(boundary_mean, (3, 1)), dtype=torch.float)
        self.assertTrue(torch.isfinite(dk_policy.log_prob_batch(state_torch[:3], boundary_batch)).all().item())

    def test_dk_legacy_log_prob_matches_old_direct_gaussian(self):
        mean = np.asarray([0.4, 1.2, 1.8, 2.1, 0.25], dtype=np.float64)
        states_torch = torch.randn(3, 7, dtype=torch.float)
        actions_torch = torch.tensor(
            [
                [0.5, 1.0, 1.5, 2.0, 0.2],
                [0.8, 1.3, 1.7, 2.2, 0.4],
                [0.3, 1.1, 1.6, 2.3, 0.6],
            ],
            dtype=torch.float,
        )
        legacy_policy = HeuristicGaussianPolicy(
            lambda state: mean,
            5,
            "cpu",
            transform_kind="mimo",
            actor_distribution="legacy",
        )

        mu = torch.tensor(np.tile(mean, (3, 1)), dtype=torch.float)
        std = torch.exp(legacy_policy.log_std).view(1, -1).expand_as(mu)
        expected = torch.distributions.normal.Normal(mu, std).log_prob(actions_torch).sum(dim=1)

        self.assertEqual(legacy_policy.actor_distribution, LEGACY_ACTOR_DISTRIBUTION)
        self.assertTrue(torch.allclose(legacy_policy.log_prob_batch(states_torch, actions_torch), expected))

        squashed_policy = HeuristicGaussianPolicy(
            lambda state: mean,
            5,
            "cpu",
            transform_kind="mimo",
            actor_distribution="squashed",
        )
        self.assertEqual(squashed_policy.actor_distribution, SQUASHED_ACTOR_DISTRIBUTION)
        self.assertFalse(torch.allclose(squashed_policy.log_prob_batch(states_torch, actions_torch), expected))

    def test_build_dk_policy_uses_legacy_actor_distribution(self):
        class FakeMimoEnv:
            UE_num = 4
            action_dim = 5

        states_torch = torch.randn(2, 7, dtype=torch.float)
        actions_torch = torch.full((2, 5), 0.5, dtype=torch.float)
        policy = _build_dk_policy("MIMO", FakeMimoEnv(), "cpu", 3, actor_distribution="legacy")

        means = []
        for idx in range(states_torch.shape[0]):
            means.append(policy.mean_action(states_torch[idx].detach().cpu().numpy()))
        mu = torch.tensor(np.asarray(means, dtype=np.float64), dtype=torch.float)
        std = torch.exp(policy.log_std).view(1, -1).expand_as(mu)
        expected = torch.distributions.normal.Normal(mu, std).log_prob(actions_torch).sum(dim=1)

        self.assertEqual(policy.actor_distribution, LEGACY_ACTOR_DISTRIBUTION)
        self.assertTrue(torch.allclose(policy.log_prob_batch(states_torch, actions_torch), expected))


if __name__ == "__main__":
    unittest.main()
