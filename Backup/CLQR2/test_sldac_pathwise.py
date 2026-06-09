import unittest

import numpy as np
import torch

from SLDAC_Pathwise import (
    POLICY_GRADIENT_MODES,
    _build_scene,
    _compute_conservative_qprop_eta,
    _compute_pathwise_gradients,
    _flatten_actor_parameters,
    _pack_pathwise_diagnostics,
)
from critic_opt import Critic
from qprop_critic import QPropCritic
from run_clqr_sldac_pathwise import build_python_config


class SldacPathwiseTest(unittest.TestCase):
    def test_run_config_exposes_pathwise_fields(self):
        config = build_python_config()
        self.assertIn("policy_gradient_mode", config)
        self.assertIn("behavior_policy_mode", config)
        self.assertIn("normalize_actor_gradient", config)
        self.assertIn("update_log_std", config)
        self.assertIn("print_actor_grad_norm", config)
        self.assertIn("save_diagnostics", config)
        self.assertIn("use_qprop_dedicated_critic", config)
        self.assertIn("qprop_critic_update_steps", config)
        self.assertIn("qprop_replay_batch_size", config)
        self.assertIn("qprop_target_action_mode", config)
        self.assertIn("qprop_critic_lr_scale", config)
        self.assertIn("qprop_target_tau_reward", config)
        self.assertIn("qprop_target_tau_cost", config)

    def test_policy_gradient_modes_include_qprop_conservative(self):
        self.assertIn("qprop_conservative", POLICY_GRADIENT_MODES)

    def test_conservative_qprop_eta_uses_positive_covariance_gate(self):
        positive_eta, positive_covariance = _compute_conservative_qprop_eta(
            torch.tensor([1.0, -1.0, 2.0, -2.0]),
            torch.tensor([0.5, -0.5, 1.0, -1.0]),
        )
        negative_eta, negative_covariance = _compute_conservative_qprop_eta(
            torch.tensor([1.0, -1.0, 2.0, -2.0]),
            torch.tensor([-0.5, 0.5, -1.0, 1.0]),
        )
        self.assertEqual(float(positive_eta), 1.0)
        self.assertGreater(float(positive_covariance), 0.0)
        self.assertEqual(float(negative_eta), 0.0)
        self.assertLess(float(negative_covariance), 0.0)

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
            torch.randn(8, action_dim, dtype=torch.float),
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
            torch.randn(8, action_dim, dtype=torch.float),
            constraint_dim,
            real_theta_dim,
            "deterministic_dpg",
            False,
            False,
        )
        self.assertEqual(tuple(grad_tilda_torch.shape), (1 + constraint_dim, real_theta_dim))
        self.assertTrue(torch.allclose(grad_tilda_torch[:, -action_dim:], torch.zeros_like(grad_tilda_torch[:, -action_dim:])))

    def test_actor_gradient_does_not_accumulate_critic_parameter_grads(self):
        _, actor, state_dim, _, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, actor.action_dim, constraint_dim, 1, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        state_batch_torch = torch.randn(8, state_dim, dtype=torch.float)
        _compute_pathwise_gradients(
            actor,
            critic,
            state_batch_torch,
            torch.randn(8, actor.action_dim, dtype=torch.float),
            constraint_dim,
            real_theta_dim,
            "stochastic_pathwise",
            False,
            True,
        )
        for head_idx in range(1 + constraint_dim):
            target_net = getattr(critic, "target_net{0}".format(head_idx))
            for param in target_net.parameters():
                self.assertIsNone(param.grad)

    def test_pathwise_gradient_can_return_saved_diagnostics(self):
        _, actor, state_dim, _, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, actor.action_dim, constraint_dim, 1, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        state_batch_torch = torch.randn(8, state_dim, dtype=torch.float)
        grad_tilda_torch, diagnostics = _compute_pathwise_gradients(
            actor,
            critic,
            state_batch_torch,
            torch.randn(8, actor.action_dim, dtype=torch.float),
            constraint_dim,
            real_theta_dim,
            "stochastic_pathwise",
            False,
            True,
            return_diagnostics=True,
        )
        self.assertEqual(tuple(grad_tilda_torch.shape), (1 + constraint_dim, real_theta_dim))
        for key in ("q_mean", "q_std", "grad_a_norm", "actor_grad_norm"):
            self.assertEqual(tuple(diagnostics[key].shape), (1 + constraint_dim,))
            self.assertTrue(np.isfinite(diagnostics[key]).all())
        self.assertEqual(tuple(diagnostics["constraint_to_objective_grad_norm_ratio"].shape), (constraint_dim,))
        self.assertTrue(np.isfinite(diagnostics["constraint_to_objective_grad_norm_ratio"]).all())

    def test_qprop_conservative_gradient_returns_diagnostics(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, action_dim, constraint_dim, 1, "cpu")
        qprop_critic = QPropCritic("CLQR", state_dim, action_dim, constraint_dim, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        state_batch_torch = torch.randn(8, state_dim, dtype=torch.float)
        action_batch_torch = torch.randn(8, action_dim, dtype=torch.float)
        grad_tilda_torch, diagnostics = _compute_pathwise_gradients(
            actor,
            critic,
            state_batch_torch,
            action_batch_torch,
            constraint_dim,
            real_theta_dim,
            "qprop_conservative",
            False,
            True,
            return_diagnostics=True,
            qprop_control_critic=qprop_critic,
            qprop_control_source_code=1.0,
        )
        self.assertEqual(tuple(grad_tilda_torch.shape), (1 + constraint_dim, real_theta_dim))
        self.assertTrue(torch.isfinite(grad_tilda_torch).all().item())
        for key in (
            "qprop_eta",
            "qprop_covariance",
            "score_grad_norm",
            "pathwise_grad_norm",
            "combined_grad_norm",
            "score_signal_mean",
            "score_signal_std",
            "control_signal_mean",
            "control_signal_std",
            "qprop_pathwise_grad_ratio",
            "qprop_score_grad_ratio",
        ):
            self.assertEqual(tuple(diagnostics[key].shape), (1 + constraint_dim,))
            self.assertTrue(np.isfinite(diagnostics[key]).all())
        self.assertEqual(float(diagnostics["qprop_control_source_code"]), 1.0)
        self.assertTrue(np.all(diagnostics["qprop_pathwise_grad_ratio"] >= 0.0))
        self.assertTrue(np.all(diagnostics["qprop_pathwise_grad_ratio"] <= 1.0))
        self.assertTrue(np.all(diagnostics["qprop_score_grad_ratio"] >= 0.0))
        self.assertTrue(np.all(diagnostics["qprop_score_grad_ratio"] <= 1.0))
        for head_idx in range(1 + constraint_dim):
            target_net = getattr(critic, "target_net{0}".format(head_idx))
            for param in target_net.parameters():
                self.assertIsNone(param.grad)
            qprop_target_net = getattr(qprop_critic, "target_net{0}".format(head_idx))
            for param in qprop_target_net.parameters():
                self.assertIsNone(param.grad)

    def test_qprop_conservative_can_use_main_critic_control_source(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, action_dim, constraint_dim, 1, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        grad_tilda_torch, diagnostics = _compute_pathwise_gradients(
            actor,
            critic,
            torch.randn(8, state_dim, dtype=torch.float),
            torch.randn(8, action_dim, dtype=torch.float),
            constraint_dim,
            real_theta_dim,
            "qprop_conservative",
            False,
            True,
            return_diagnostics=True,
            qprop_control_critic=None,
            qprop_control_source_code=0.0,
        )
        self.assertEqual(tuple(grad_tilda_torch.shape), (1 + constraint_dim, real_theta_dim))
        self.assertEqual(float(diagnostics["qprop_control_source_code"]), 0.0)

    def test_qprop_conservative_can_freeze_log_std(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 4)
        critic = Critic("CLQR", 4, state_dim, action_dim, constraint_dim, 1, "cpu")
        _, real_theta_dim = _flatten_actor_parameters(actor)
        grad_tilda_torch = _compute_pathwise_gradients(
            actor,
            critic,
            torch.randn(8, state_dim, dtype=torch.float),
            torch.randn(8, action_dim, dtype=torch.float),
            constraint_dim,
            real_theta_dim,
            "qprop_conservative",
            False,
            False,
        )
        self.assertTrue(torch.allclose(grad_tilda_torch[:, -action_dim:], torch.zeros_like(grad_tilda_torch[:, -action_dim:])))

    def test_pack_pathwise_diagnostics_for_mat_payload(self):
        history = [
            {
                "update_index": 1,
                "global_step": 1001,
                "episode_index": 0,
                "q_mean": np.array([1.0, 2.0]),
                "q_std": np.array([0.1, 0.2]),
                "grad_a_norm": np.array([4.0, 5.0]),
                "actor_grad_norm": np.array([7.0, 8.0]),
                "constraint_to_objective_grad_norm_ratio": np.array([8.0 / 7.0]),
                "score_grad_norm": np.array([1.0, 2.0]),
                "pathwise_grad_norm": np.array([4.0, 5.0]),
                "combined_grad_norm": np.array([7.0, 8.0]),
                "qprop_eta": np.array([1.0, 0.0]),
                "qprop_covariance": np.array([0.3, -0.2]),
                "score_signal_mean": np.array([0.0, 0.1]),
                "score_signal_std": np.array([1.0, 1.1]),
                "control_signal_mean": np.array([0.0, 0.2]),
                "control_signal_std": np.array([0.5, 0.6]),
                "qprop_critic_loss": np.array([0.1, 0.2]),
                "qprop_critic_td_error_mean": np.array([0.4, 0.5]),
                "qprop_critic_target_mean": np.array([0.7, 0.8]),
                "qprop_critic_pred_mean": np.array([1.0, 1.1]),
                "qprop_replay_batch_size": 8.0,
                "qprop_control_source_code": 1.0,
                "qprop_target_action_mode_code": 0.0,
                "qprop_pathwise_grad_ratio": np.array([0.8, 5.0 / 7.0]),
                "qprop_score_grad_ratio": np.array([0.2, 2.0 / 7.0]),
            }
        ]
        payload = _pack_pathwise_diagnostics(history, constraint_dim=1)
        self.assertEqual(tuple(payload["q_mean"].shape), (1, 2))
        self.assertEqual(tuple(payload["constraint_to_objective_grad_norm_ratio"].shape), (1, 1))
        self.assertEqual(tuple(payload["qprop_eta"].shape), (1, 2))
        self.assertEqual(tuple(payload["combined_grad_norm"].shape), (1, 2))
        self.assertEqual(tuple(payload["qprop_critic_loss"].shape), (1, 2))
        self.assertEqual(tuple(payload["qprop_pathwise_grad_ratio"].shape), (1, 2))
        self.assertEqual(payload["qprop_replay_batch_size"].tolist(), [8.0])
        self.assertEqual(payload["qprop_control_source_code"].tolist(), [1.0])
        self.assertEqual(payload["update_index"].tolist(), [1])
        self.assertEqual(payload["global_step"].tolist(), [1001])
        self.assertEqual(payload["episode_index"].tolist(), [0])


if __name__ == "__main__":
    unittest.main()
