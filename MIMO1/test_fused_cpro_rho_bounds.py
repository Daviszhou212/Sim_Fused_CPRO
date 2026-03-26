import unittest

import numpy as np
import torch

from critic_opt import Critic
from Fused_CPRO import (
    _blend_online_offline_loss,
    _build_rho_lower_bounds,
    _estimate_mixed_func_value_tilda,
    _normalize_old_policy_sampling_probs,
    _normalize_simplex,
    _select_policy_gradient_batch_impl,
)


class _FakeCriticNet:
    def __init__(self, value):
        self.value = float(value)

    def forward(self, state_batch_torch, action_batch_torch):
        batch_size = int(state_batch_torch.shape[0])
        return torch.full((batch_size, 1), self.value, dtype=torch.float, device=state_batch_torch.device)


def _build_offline_dataset(state_shape=(4, 2), action_shape=(4, 1), cost_dim=5, state_value=9.0, action_value=-7.0, cost_value=5.0):
    return {
        "state": np.full(state_shape, state_value, dtype=np.float64),
        "action": np.full(action_shape, action_value, dtype=np.float64),
        "costs": np.full((state_shape[0], cost_dim), cost_value, dtype=np.float64),
    }


class FusedCproRhoBoundsTest(unittest.TestCase):
    def test_new_actor_lower_bound_is_first_dimension(self):
        rho_lower_bounds = _build_rho_lower_bounds(4)
        self.assertAlmostEqual(float(rho_lower_bounds[0]), 0.2)
        np.testing.assert_allclose(rho_lower_bounds[1:], np.full((3,), 1e-4, dtype=np.float64))

    def test_normalize_simplex_enforces_new_actor_floor(self):
        rho_lower_bounds = _build_rho_lower_bounds(4)
        rho = _normalize_simplex(np.asarray([0.0, 1.0, 1.0, 1.0], dtype=np.float64), rho_lower_bounds)
        self.assertAlmostEqual(float(np.sum(rho)), 1.0, places=10)
        self.assertAlmostEqual(float(rho[0]), 0.2, places=10)
        self.assertTrue(np.all(rho >= rho_lower_bounds - 1e-12))

    def test_infeasible_lower_bounds_raise_value_error(self):
        with self.assertRaises(ValueError):
            _build_rho_lower_bounds(5, rho_min_new_actor=0.8, rho_min_old_policy=0.1)

    def test_old_policy_sampling_probs_are_renormalized_from_old_slice(self):
        probs = _normalize_old_policy_sampling_probs(np.asarray([0.2, 0.3, 0.5], dtype=np.float64), old_policy_count=2)
        np.testing.assert_allclose(probs, np.asarray([0.375, 0.625], dtype=np.float64))

    def test_old_policy_sampling_probs_fall_back_to_uniform_when_old_mass_is_zero(self):
        probs = _normalize_old_policy_sampling_probs(np.asarray([1.0, 0.0, 0.0], dtype=np.float64), old_policy_count=2)
        np.testing.assert_allclose(probs, np.asarray([0.5, 0.5], dtype=np.float64))

    def test_policy_gradient_batch_keeps_equal_online_and_offline_size_when_xi_is_zero(self):
        online_state_batch = np.arange(12, dtype=np.float64).reshape(6, 2)
        online_action_batch = np.arange(6, dtype=np.float64).reshape(6, 1)
        offline_datasets = [_build_offline_dataset()]
        fused_state_batch, fused_action_batch = _select_policy_gradient_batch_impl(
            online_state_batch,
            online_action_batch,
            offline_datasets,
            xi=0.0,
            grad_t=4,
            state_dim=2,
            action_dim=1,
            use_offline_data=True,
        )
        self.assertEqual(fused_state_batch.shape, (8, 2))
        self.assertEqual(fused_action_batch.shape, (8, 1))
        np.testing.assert_allclose(fused_state_batch[:4], online_state_batch[-4:])
        np.testing.assert_allclose(fused_action_batch[:4], online_action_batch[-4:])
        np.testing.assert_allclose(fused_state_batch[4:], np.full((4, 2), 9.0))
        np.testing.assert_allclose(fused_action_batch[4:], np.full((4, 1), -7.0))

    def test_policy_gradient_batch_size_does_not_change_with_xi(self):
        online_state_batch = np.arange(12, dtype=np.float64).reshape(6, 2)
        online_action_batch = np.arange(6, dtype=np.float64).reshape(6, 1)
        offline_datasets = [_build_offline_dataset()]
        fused_state_batch, fused_action_batch = _select_policy_gradient_batch_impl(
            online_state_batch,
            online_action_batch,
            offline_datasets,
            xi=0.5,
            grad_t=4,
            state_dim=2,
            action_dim=1,
            use_offline_data=True,
        )
        self.assertEqual(fused_state_batch.shape, (8, 2))
        self.assertEqual(fused_action_batch.shape, (8, 1))
        np.testing.assert_allclose(fused_state_batch[:4], online_state_batch[-4:])
        np.testing.assert_allclose(fused_action_batch[:4], online_action_batch[-4:])
        np.testing.assert_allclose(fused_state_batch[4:], np.full((4, 2), 9.0))
        np.testing.assert_allclose(fused_action_batch[4:], np.full((4, 1), -7.0))

    def test_policy_gradient_batch_keeps_equal_online_and_offline_size_when_xi_is_one(self):
        online_state_batch = np.arange(12, dtype=np.float64).reshape(6, 2)
        online_action_batch = np.arange(6, dtype=np.float64).reshape(6, 1)
        offline_datasets = [_build_offline_dataset()]
        fused_state_batch, fused_action_batch = _select_policy_gradient_batch_impl(
            online_state_batch,
            online_action_batch,
            offline_datasets,
            xi=1.0,
            grad_t=4,
            state_dim=2,
            action_dim=1,
            use_offline_data=True,
        )
        self.assertEqual(fused_state_batch.shape, (8, 2))
        self.assertEqual(fused_action_batch.shape, (8, 1))
        np.testing.assert_allclose(fused_state_batch[:4], online_state_batch[-4:])
        np.testing.assert_allclose(fused_action_batch[:4], online_action_batch[-4:])
        np.testing.assert_allclose(fused_state_batch[4:], np.full((4, 2), 9.0))
        np.testing.assert_allclose(fused_action_batch[4:], np.full((4, 1), -7.0))

    def test_blend_online_offline_loss_uses_convex_combination(self):
        blended = _blend_online_offline_loss(2.0, 6.0, xi=0.25, use_offline_data=True)
        self.assertAlmostEqual(float(blended), 3.0)

    def test_blend_online_offline_loss_returns_online_loss_when_offline_is_disabled(self):
        blended = _blend_online_offline_loss(2.0, 6.0, xi=0.75, use_offline_data=False)
        self.assertAlmostEqual(float(blended), 2.0)

    def test_mixed_func_value_uses_convex_combination(self):
        online_costs = np.asarray(
            [
                [1.0, 3.0, 5.0, 7.0, 9.0],
                [3.0, 5.0, 7.0, 9.0, 11.0],
            ],
            dtype=np.float64,
        )
        offline_datasets = [
            {
                "state": np.zeros((2, 2), dtype=np.float64),
                "action": np.zeros((2, 1), dtype=np.float64),
                "costs": np.full((2, 5), [10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            }
        ]
        mixed = _estimate_mixed_func_value_tilda(online_costs, offline_datasets, xi=0.25, use_offline_data=True)
        np.testing.assert_allclose(mixed, np.asarray([4.0, 8.0, 12.0, 16.0, 20.0], dtype=np.float64))

    def test_mixed_func_value_returns_online_mean_when_offline_is_disabled(self):
        online_costs = np.asarray(
            [
                [1.0, 3.0, 5.0, 7.0, 9.0],
                [3.0, 5.0, 7.0, 9.0, 11.0],
            ],
            dtype=np.float64,
        )
        mixed = _estimate_mixed_func_value_tilda(online_costs, offline_datasets=[], xi=1.0, use_offline_data=False)
        np.testing.assert_allclose(mixed, np.asarray([2.0, 4.0, 6.0, 8.0, 10.0], dtype=np.float64))

    def test_mixed_func_value_returns_offline_mean_when_xi_is_one(self):
        online_costs = np.asarray(
            [
                [1.0, 3.0, 5.0, 7.0, 9.0],
                [3.0, 5.0, 7.0, 9.0, 11.0],
            ],
            dtype=np.float64,
        )
        offline_datasets = [
            {
                "state": np.zeros((2, 2), dtype=np.float64),
                "action": np.zeros((2, 1), dtype=np.float64),
                "costs": np.full((2, 5), [10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64),
            }
        ]
        mixed = _estimate_mixed_func_value_tilda(online_costs, offline_datasets, xi=1.0, use_offline_data=True)
        np.testing.assert_allclose(mixed, np.asarray([10.0, 20.0, 30.0, 40.0, 50.0], dtype=np.float64))

    def test_critic_value_supports_dynamic_batch_size(self):
        critic = Critic.__new__(Critic)
        critic.example_name = "MIMO"
        critic.constraint_dim = 4
        critic.device = "cpu"
        critic.num_new_data = 4
        critic.target_net0 = _FakeCriticNet(1.0)
        critic.target_net1 = _FakeCriticNet(2.0)
        critic.target_net2 = _FakeCriticNet(3.0)
        critic.target_net3 = _FakeCriticNet(4.0)
        critic.target_net4 = _FakeCriticNet(5.0)
        state_batch_torch = torch.zeros((7, 3), dtype=torch.float)
        action_batch_torch = torch.zeros((7, 2), dtype=torch.float)

        q_hat_torch = critic.critic_value(state_batch_torch, action_batch_torch)

        self.assertEqual(tuple(q_hat_torch.shape), (7, 5))

    def test_policy_gradient_batch_ignores_xi_when_offline_data_is_disabled(self):
        online_state_batch = np.arange(12, dtype=np.float64).reshape(6, 2)
        online_action_batch = np.arange(6, dtype=np.float64).reshape(6, 1)
        fused_state_batch, fused_action_batch = _select_policy_gradient_batch_impl(
            online_state_batch,
            online_action_batch,
            offline_datasets=[],
            xi=1.0,
            grad_t=4,
            state_dim=2,
            action_dim=1,
            use_offline_data=False,
        )
        np.testing.assert_allclose(fused_state_batch, online_state_batch[-4:])
        np.testing.assert_allclose(fused_action_batch, online_action_batch[-4:])

    def test_policy_gradient_batch_can_use_old_policy_sampling_probs(self):
        online_state_batch = np.arange(12, dtype=np.float64).reshape(6, 2)
        online_action_batch = np.arange(6, dtype=np.float64).reshape(6, 1)
        offline_datasets = [
            _build_offline_dataset(state_value=9.0, action_value=-7.0),
            _build_offline_dataset(state_value=13.0, action_value=-11.0),
        ]
        fused_state_batch, fused_action_batch = _select_policy_gradient_batch_impl(
            online_state_batch,
            online_action_batch,
            offline_datasets,
            xi=0.5,
            grad_t=4,
            state_dim=2,
            action_dim=1,
            use_offline_data=True,
            dataset_probs=np.asarray([0.0, 1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(fused_state_batch[4:], np.full((4, 2), 13.0))
        np.testing.assert_allclose(fused_action_batch[4:], np.full((4, 1), -11.0))

    def test_mixed_func_value_can_use_old_policy_sampling_probs(self):
        online_costs = np.asarray(
            [
                [1.0, 3.0, 5.0, 7.0, 9.0],
                [3.0, 5.0, 7.0, 9.0, 11.0],
            ],
            dtype=np.float64,
        )
        offline_datasets = [
            _build_offline_dataset(cost_dim=5, cost_value=[10.0, 20.0, 30.0, 40.0, 50.0]),
            _build_offline_dataset(cost_dim=5, cost_value=[30.0, 40.0, 50.0, 60.0, 70.0]),
        ]
        mixed = _estimate_mixed_func_value_tilda(
            online_costs,
            offline_datasets,
            xi=1.0,
            use_offline_data=True,
            dataset_probs=np.asarray([0.0, 1.0], dtype=np.float64),
        )
        np.testing.assert_allclose(mixed, np.asarray([30.0, 40.0, 50.0, 60.0, 70.0], dtype=np.float64))


if __name__ == "__main__":
    unittest.main()
