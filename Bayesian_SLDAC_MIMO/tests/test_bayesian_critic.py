import unittest

import numpy as np
import torch


class BayesianCriticTest(unittest.TestCase):
    def test_buffer_and_critic_shapes(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import BayesianCritic
        from Bayesian_SLDAC_MIMO.buffer import DataStorage

        storage = DataStorage(T=4, num_new_data=2, state_dim=3, action_dim=2, constraint_dim=1, window=5, q=1)
        for idx in range(8):
            storage.store_experiences(
                np.full(3, idx, dtype=np.float64),
                np.full(2, idx, dtype=np.float64),
                np.array([idx, idx - 0.5], dtype=np.float64),
                np.full(3, idx + 1, dtype=np.float64),
                float(idx),
                float(idx) / 2,
            )

        state, action, costs, next_state, reward, cost = storage.take_experiences()
        self.assertEqual(state.shape, (8, 3))
        self.assertEqual(action.shape, (8, 2))
        self.assertEqual(costs.shape, (8, 2))
        self.assertEqual(next_state.shape, (8, 3))
        self.assertEqual(reward.shape[1], 1)
        self.assertEqual(cost.shape[1], 1)

        critic = BayesianCritic("MIMO", num_new_data=4, state_dim=3, action_dim=2, constraint_dim=1, q=1, device="cpu", ensemble_size=3)
        states_torch = torch.zeros((4, 3), dtype=torch.float32)
        actions_torch = torch.zeros((4, 2), dtype=torch.float32)
        q_mean, q_std = critic.critic_value_stats(states_torch, actions_torch)

        self.assertEqual(tuple(q_mean.shape), (4, 2))
        self.assertEqual(tuple(q_std.shape), (4, 2))
        self.assertTrue(torch.isfinite(q_mean).all().item())
        self.assertTrue(torch.isfinite(q_std).all().item())

    def test_critic_lr_base_controls_optimizer_lr(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import BayesianCritic

        critic = BayesianCritic(
            "MIMO",
            num_new_data=4,
            state_dim=3,
            action_dim=2,
            constraint_dim=1,
            q=4,
            device="cpu",
            ensemble_size=1,
            critic_lr_base=0.02,
        )

        lr = critic.optimizers[0][0].param_groups[0]["lr"]
        self.assertAlmostEqual(lr, 0.01)

    def test_risk_correction_and_q_hat_normalization(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import normalize_q_hat_like_sldac_code, risk_correct_q_values

        q_mean = np.array([[10.0, -0.2], [12.0, 0.0], [14.0, 0.2]], dtype=np.float64)
        q_std = np.array([[2.0, 0.4], [4.0, 0.3], [6.0, 0.2]], dtype=np.float64)

        no_risk = risk_correct_q_values(q_mean, q_std, beta_uncertainty=0.0)
        np.testing.assert_allclose(no_risk, q_mean)

        corrected = risk_correct_q_values(q_mean, q_std, beta_uncertainty=0.5)
        np.testing.assert_allclose(corrected[:, 0], np.array([9.0, 10.0, 11.0]))
        np.testing.assert_allclose(corrected[:, 1], np.array([0.0, 0.15, 0.3]))

        q_hat = normalize_q_hat_like_sldac_code(corrected)
        self.assertEqual(q_hat.shape, corrected.shape)
        self.assertAlmostEqual(float(np.mean(q_hat[:, 0])), 0.0, places=6)
        self.assertAlmostEqual(float(np.mean(q_hat[:, 1])), 0.0, places=6)

    def test_q_hat_normalization_uses_mutated_objective_std_like_sldac_code(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import normalize_q_hat_like_sldac_code

        q_values = np.array(
            [
                [1.0, 10.0],
                [3.0, 20.0],
                [5.0, 40.0],
            ],
            dtype=np.float64,
        )
        expected = q_values.copy()
        expected[:, 0] = (expected[:, 0] - np.mean(expected[:, 0])) / (np.std(expected[:, 0]) + 1e-6)
        expected[:, 1] = (expected[:, 1] - np.mean(expected[:, 1])) / (np.std(expected[:, 0]) + 1e-6)

        actual = normalize_q_hat_like_sldac_code(q_values)

        np.testing.assert_allclose(actual, expected)

    def test_log_std_gradient_accumulates_across_heads_like_sldac_code(self):
        from Bayesian_SLDAC_MIMO.model import GaussianPolicy_MIMO, flatten_actor_grad
        from Bayesian_SLDAC_MIMO.sldac import _estimate_actor_gradient_rows

        torch.manual_seed(0)
        actor = GaussianPolicy_MIMO(3, 2, "cpu", 4)
        state_batch = torch.zeros((4, 3), dtype=torch.float32)
        action_batch = torch.zeros((4, 2), dtype=torch.float32)
        q_hat = torch.tensor(
            [
                [1.0, 0.5],
                [2.0, 1.5],
                [3.0, 2.5],
                [4.0, 3.5],
            ],
            dtype=torch.float32,
        )

        actor.zero_grad()
        (q_hat[:, 0] * actor.evaluate_action(state_batch, action_batch)).mean().backward()
        first_grad = flatten_actor_grad(actor)[-actor.action_dim :].copy()
        actor.zero_grad()
        (q_hat[:, 1] * actor.evaluate_action(state_batch, action_batch)).mean().backward()
        accumulated_grad = flatten_actor_grad(actor)[-actor.action_dim :].copy()

        torch.manual_seed(0)
        isolated_actor = GaussianPolicy_MIMO(3, 2, "cpu", 4)
        isolated_actor.zero_grad()
        (q_hat[:, 1] * isolated_actor.evaluate_action(state_batch, action_batch)).mean().backward()
        isolated_grad = flatten_actor_grad(isolated_actor)[-isolated_actor.action_dim :].copy()

        np.testing.assert_allclose(accumulated_grad, first_grad + isolated_grad)

        torch.manual_seed(0)
        helper_actor = GaussianPolicy_MIMO(3, 2, "cpu", 4)
        helper_grad = _estimate_actor_gradient_rows(helper_actor, q_hat, state_batch, action_batch, constraint_dim=1)
        np.testing.assert_allclose(helper_grad[1, -helper_actor.action_dim :], accumulated_grad)

    def test_ensemble_initialization_preserves_actor_sampling_stream(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import BayesianCritic
        from Bayesian_SLDAC_MIMO.model import GaussianPolicy_MIMO

        def first_action_after_critic(ensemble_size, ensemble_init_mode="shared"):
            torch.manual_seed(0)
            actor = GaussianPolicy_MIMO(3, 2, "cpu", 4)
            BayesianCritic(
                "MIMO",
                num_new_data=4,
                state_dim=3,
                action_dim=2,
                constraint_dim=1,
                q=1,
                device="cpu",
                ensemble_size=ensemble_size,
                ensemble_init_mode=ensemble_init_mode,
            )
            return actor.sample_action(np.array([0.1, -0.2, 0.3], dtype=np.float64))

        legacy_action = first_action_after_critic(1)
        bayesian_action = first_action_after_critic(5)

        np.testing.assert_allclose(bayesian_action, legacy_action)

    def test_shared_ensemble_initialization_starts_members_identical(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import BayesianCritic

        torch.manual_seed(0)
        critic = BayesianCritic(
            "MIMO",
            num_new_data=4,
            state_dim=3,
            action_dim=2,
            constraint_dim=1,
            q=1,
            device="cpu",
            ensemble_size=3,
            ensemble_init_mode="shared",
        )
        states_torch = torch.zeros((4, 3), dtype=torch.float32)
        actions_torch = torch.zeros((4, 2), dtype=torch.float32)

        _q_mean, q_std = critic.critic_value_stats(states_torch, actions_torch)

        self.assertTrue(torch.allclose(q_std, torch.zeros_like(q_std)))

    def test_independent_ensemble_initialization_starts_members_distinct(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import BayesianCritic

        torch.manual_seed(0)
        critic = BayesianCritic(
            "MIMO",
            num_new_data=4,
            state_dim=3,
            action_dim=2,
            constraint_dim=1,
            q=1,
            device="cpu",
            ensemble_size=3,
            ensemble_init_mode="independent",
        )
        states_torch = torch.zeros((4, 3), dtype=torch.float32)
        actions_torch = torch.zeros((4, 2), dtype=torch.float32)

        _q_mean, q_std = critic.critic_value_stats(states_torch, actions_torch)

        self.assertGreater(float(torch.sum(q_std).item()), 0.0)

    def test_bootstrap_masks_do_not_consume_actor_sampling_stream(self):
        from Bayesian_SLDAC_MIMO.bayesian_critic import BayesianCritic
        from Bayesian_SLDAC_MIMO.model import GaussianPolicy_MIMO

        def action_after_update(ensemble_size):
            torch.manual_seed(0)
            actor = GaussianPolicy_MIMO(3, 2, "cpu", 4)
            critic = BayesianCritic(
                "MIMO",
                num_new_data=4,
                state_dim=3,
                action_dim=2,
                constraint_dim=1,
                q=1,
                device="cpu",
                ensemble_size=ensemble_size,
                bootstrap_mask_prob=0.8,
            )
            state = np.array([0.1, -0.2, 0.3], dtype=np.float64)
            _ = actor.sample_action(state)
            state_batch = np.zeros((4, 3), dtype=np.float64)
            action_batch = np.zeros((4, 2), dtype=np.float64)
            costs_batch = np.zeros((4, 2), dtype=np.float64)
            next_state_batch = np.zeros((4, 3), dtype=np.float64)
            next_action_batch = np.zeros((4, 2), dtype=np.float64)
            critic.critic_update(
                np.zeros(2, dtype=np.float64),
                state_batch,
                action_batch,
                costs_batch,
                next_state_batch,
                next_action_batch,
                eta=1.0,
                gamma_reward=0.5,
                gamma_cost=0.5,
            )
            return actor.sample_action(state)

        legacy_action = action_after_update(1)
        bayesian_action = action_after_update(5)

        np.testing.assert_allclose(bayesian_action, legacy_action)


if __name__ == "__main__":
    unittest.main()
