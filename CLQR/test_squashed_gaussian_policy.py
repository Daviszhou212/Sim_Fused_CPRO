import unittest

import numpy as np
import torch

from Fused_CPRO import (
    HeuristicGaussianPolicy,
    _build_mixture_log_prob,
    _log_prob_batch,
)
from model import (
    CLQR_ACTION_MAX,
    GaussianPolicy_CLQR,
    clqr_inverse_action_and_log_det,
)


class SquashedGaussianClqrTest(unittest.TestCase):
    def test_clqr_actor_bounds_and_manual_log_prob(self):
        actor = GaussianPolicy_CLQR(state_dim=15, action_dim=4, device="cpu", num_new_data=4)
        state_torch = torch.randn(6, 15, dtype=torch.float)
        action_torch = actor.sample_action_tensor(
            state_torch,
            reparameterized=True,
            use_mean=False,
            track_log_std_grad=True,
        )

        self.assertTrue(torch.all(action_torch > -CLQR_ACTION_MAX).item())
        self.assertTrue(torch.all(action_torch < CLQR_ACTION_MAX).item())
        self.assertTrue(torch.isfinite(actor.evaluate_action(state_torch, action_torch)).all().item())

        raw_loc = actor.net(state_torch)
        raw_action, log_det, valid = clqr_inverse_action_and_log_det(action_torch)
        expected = torch.distributions.normal.Normal(
            raw_loc,
            torch.exp(actor.log_std).view(1, -1).expand_as(raw_loc),
        ).log_prob(raw_action).sum(dim=1) - log_det
        actual = actor.evaluate_action_with_log_std(state_torch, action_torch, actor.log_std)

        self.assertTrue(valid.all().item())
        self.assertTrue(torch.allclose(actual, expected, atol=1e-5, rtol=1e-5))

        boundary_action = action_torch.detach().clone()
        boundary_action[:, 0] = CLQR_ACTION_MAX
        self.assertTrue(torch.isneginf(actor.evaluate_action(state_torch, boundary_action)).all().item())

    def test_fused_mixture_keeps_logsumexp_and_log_std_grad(self):
        actor = GaussianPolicy_CLQR(state_dim=15, action_dim=4, device="cpu", num_new_data=4)
        state_torch = torch.randn(5, 15, dtype=torch.float)
        action_torch = actor.sample_action_tensor(
            state_torch,
            reparameterized=False,
            use_mean=False,
            track_log_std_grad=False,
        ).detach()

        boundary_mean = np.asarray([CLQR_ACTION_MAX, -CLQR_ACTION_MAX, 0.0, CLQR_ACTION_MAX], dtype=np.float64)
        dk_policy = HeuristicGaussianPolicy(lambda state: boundary_mean, 4, "cpu", transform_kind="clqr")
        rho = np.asarray([0.65, 0.35], dtype=np.float64)
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

        sampled = dk_policy.sample_action(np.zeros((15,), dtype=np.float64))
        np.testing.assert_allclose(sampled, boundary_mean)
        boundary_batch = torch.tensor(np.tile(boundary_mean, (3, 1)), dtype=torch.float)
        self.assertTrue(torch.isfinite(dk_policy.log_prob_batch(state_torch[:3], boundary_batch)).all().item())


if __name__ == "__main__":
    unittest.main()
