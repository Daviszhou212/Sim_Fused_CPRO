import unittest

import torch

from SLDAC_Pathwise import (
    _build_scene,
    _compute_pathwise_gradients,
    _flatten_actor_parameters,
)
from critic_opt import Critic
from run_clqr_sldac_pathwise import build_python_config


class SldacPathwiseTest(unittest.TestCase):
    def test_run_config_exposes_pathwise_fields(self):
        config = build_python_config()
        self.assertIn("policy_gradient_mode", config)
        self.assertIn("behavior_policy_mode", config)
        self.assertIn("normalize_actor_gradient", config)
        self.assertIn("update_log_std", config)

    def test_actor_exposes_mean_and_reparameterized_sampling(self):
        _, actor, state_dim, action_dim, _, _ = _build_scene("CLQR", 0, "cpu", 4)
        state_batch_torch = torch.randn(6, state_dim, dtype=torch.float)
        mean_action = actor.mean_action_tensor(state_batch_torch)
        sample_action = actor.sample_action_tensor(
            state_batch_torch,
            reparameterized=True,
            use_mean=False,
            track_log_std_grad=True,
        )
        self.assertEqual(tuple(mean_action.shape), (6, action_dim))
        self.assertEqual(tuple(sample_action.shape), (6, action_dim))
        self.assertTrue(torch.isfinite(sample_action).all().item())

    def test_stochastic_pathwise_gradient_shape(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, action_dim, constraint_dim, 1, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        state_batch_torch = torch.randn(8, state_dim, dtype=torch.float)
        grad_tilda_torch = _compute_pathwise_gradients(
            actor,
            critic,
            state_batch_torch,
            constraint_dim,
            real_theta_dim,
            "stochastic_pathwise",
            True,
            True,
        )
        self.assertEqual(tuple(grad_tilda_torch.shape), (1 + constraint_dim, real_theta_dim))
        self.assertTrue(torch.isfinite(grad_tilda_torch).all().item())

    def test_deterministic_pathwise_can_freeze_log_std(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, action_dim, constraint_dim, 1, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        state_batch_torch = torch.randn(8, state_dim, dtype=torch.float)
        grad_tilda_torch = _compute_pathwise_gradients(
            actor,
            critic,
            state_batch_torch,
            constraint_dim,
            real_theta_dim,
            "deterministic_dpg",
            False,
            False,
        )
        self.assertEqual(tuple(grad_tilda_torch.shape), (1 + constraint_dim, real_theta_dim))
        self.assertTrue(torch.allclose(grad_tilda_torch[:, -action_dim:], torch.zeros_like(grad_tilda_torch[:, -action_dim:])))


if __name__ == "__main__":
    unittest.main()
