import unittest

import numpy as np
import torch

from qprop_critic import QPropCritic
from SLDAC_Pathwise import _build_scene


def _make_replay(batch_size, state_dim, action_dim, head_count):
    state = np.linspace(-0.5, 0.5, batch_size * state_dim, dtype=np.float64).reshape(batch_size, state_dim)
    action = np.linspace(-0.25, 0.25, batch_size * action_dim, dtype=np.float64).reshape(batch_size, action_dim)
    costs = np.zeros((batch_size, head_count), dtype=np.float64)
    costs[:, 0] = np.arange(batch_size, dtype=np.float64) + 10.0
    for head_idx in range(1, head_count):
        costs[:, head_idx] = head_idx + 0.1 * np.arange(batch_size, dtype=np.float64)
    next_state = state + 0.01
    return state, action, costs, next_state


def _zero_module(module):
    for param in module.parameters():
        param.data.zero_()


class QPropCriticTest(unittest.TestCase):
    def test_mimo_head_count_and_invalid_constraint_dim(self):
        critic = QPropCritic("MIMO", 69, 5, 4, "cpu")
        self.assertEqual(critic.head_count, 5)
        with self.assertRaises(ValueError):
            QPropCritic("MIMO", 69, 5, 1, "cpu")

    def test_update_from_replay_returns_finite_diagnostics_and_detaches_actor(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("MIMO", 0, "cpu", 4)
        critic = QPropCritic("MIMO", state_dim, action_dim, constraint_dim, "cpu")
        state, action, costs, next_state = _make_replay(8, state_dim, action_dim, 1 + constraint_dim)

        for head_idx in range(critic.head_count):
            _zero_module(getattr(critic, "target_net{0}".format(head_idx)))

        diagnostics = critic.update_from_replay(
            func_value=np.zeros(1 + constraint_dim, dtype=np.float64),
            state_buffer=state,
            action_buffer=action,
            costs_buffer=costs,
            next_state_buffer=next_state,
            actor=actor,
            batch_size=8,
            update_steps=1,
            target_action_mode="mean",
            tau_reward=0.0,
            tau_cost=0.0,
            rng=np.random.RandomState(0),
        )

        for key in (
            "qprop_critic_loss",
            "qprop_critic_td_error_mean",
            "qprop_critic_target_mean",
            "qprop_critic_pred_mean",
        ):
            self.assertEqual(tuple(diagnostics[key].shape), (1 + constraint_dim,))
            self.assertTrue(np.isfinite(diagnostics[key]).all())
        self.assertEqual(float(diagnostics["qprop_replay_batch_size"]), 8.0)
        self.assertEqual(float(diagnostics["qprop_control_source_code"]), 1.0)
        self.assertEqual(float(diagnostics["qprop_target_action_mode_code"]), 0.0)
        self.assertAlmostEqual(
            float(diagnostics["qprop_critic_target_mean"][0]),
            float(np.mean(costs[:, 0])),
            places=5,
        )
        for param in actor.net.parameters():
            self.assertIsNone(param.grad)
        self.assertIsNone(getattr(actor.log_std, "grad", None))

    def test_target_head_value_keeps_action_gradient(self):
        critic = QPropCritic("MIMO", 69, 5, 4, "cpu")
        state = torch.randn(6, 69, dtype=torch.float)
        action = torch.randn(6, 5, dtype=torch.float, requires_grad=True)
        value = critic.head_value(0, state, action, use_target=True)
        self.assertEqual(tuple(value.shape), (6,))
        value.sum().backward()
        self.assertIsNotNone(action.grad)
        self.assertTrue(torch.isfinite(action.grad).all().item())


if __name__ == "__main__":
    unittest.main()
