import unittest

import numpy as np
import torch


class ModelTest(unittest.TestCase):
    def test_shared_actor_action_support_and_shapes(self):
        from MultiCell_MIMO.model import SharedLocalGaussianActor

        actor = SharedLocalGaussianActor(
            local_state_dim=5,
            users_per_cell=2,
            cell_count=3,
            hidden_dims=(16,),
            device="cpu",
        )
        local_states = np.zeros((3, 5), dtype=np.float64)
        action = actor.sample_action(local_states, use_mean=True)
        cell_action = actor.sample_cell_action(local_states[1], use_mean=True)

        self.assertEqual(action.shape, (9,))
        self.assertEqual(cell_action.shape, (3,))
        self.assertTrue(np.all(action[:2] > 0.0))
        self.assertTrue(np.all(action[:2] < actor.power_max))
        self.assertTrue(np.all(action[2::3] > 0.0))

    def test_joint_log_prob_equals_sum_of_cell_log_probs(self):
        from MultiCell_MIMO.model import SharedLocalGaussianActor

        actor = SharedLocalGaussianActor(
            local_state_dim=5,
            users_per_cell=1,
            cell_count=2,
            hidden_dims=(8,),
            device="cpu",
        )
        local_states = torch.zeros((1, 2, 5), dtype=torch.float32)
        action = actor.sample_action_tensor(local_states, use_mean=False)

        joint_log_prob = actor.evaluate_action(local_states, action)
        cell_log_probs = actor.evaluate_cells(local_states, action)

        self.assertEqual(joint_log_prob.shape, (1,))
        self.assertTrue(torch.isfinite(joint_log_prob).all().item())
        self.assertTrue(torch.allclose(joint_log_prob, cell_log_probs.sum(dim=1), atol=1e-5))

    def test_log_prob_uses_direct_bounded_action_gaussian(self):
        from MultiCell_MIMO.model import SharedLocalGaussianActor

        actor = SharedLocalGaussianActor(
            local_state_dim=5,
            users_per_cell=1,
            cell_count=2,
            hidden_dims=(8,),
            device="cpu",
        )
        local_states = torch.zeros((3, 2, 5), dtype=torch.float32)
        action = torch.full((3, 4), 0.75, dtype=torch.float32)

        mean = actor._raw_mean_batch(local_states)
        std = torch.exp(actor.log_std).view(1, actor.cell_count, actor.cell_action_dim)
        expected = torch.distributions.Normal(mean, std).log_prob(
            action.view(3, 2, actor.cell_action_dim)
        ).sum(dim=(1, 2))

        actual = actor.evaluate_action(local_states, action)

        self.assertTrue(torch.allclose(actual, expected, atol=1e-5))

    def test_flatten_restore_round_trip(self):
        from MultiCell_MIMO.model import SharedLocalGaussianActor

        actor = SharedLocalGaussianActor(
            local_state_dim=4,
            users_per_cell=1,
            cell_count=2,
            hidden_dims=(8,),
            device="cpu",
        )
        original = actor.flatten_parameters().detach().clone()
        updated = original + 0.01
        actor.restore_parameters(updated)
        self.assertTrue(torch.allclose(actor.flatten_parameters(), updated))

    def test_log_std_matches_sldac_code_manual_tensor_mechanism(self):
        from MultiCell_MIMO.model import SharedLocalGaussianActor

        actor = SharedLocalGaussianActor(
            local_state_dim=4,
            users_per_cell=1,
            cell_count=2,
            hidden_dims=(8,),
            device="cpu",
        )

        parameter_ids = {id(param) for param in actor.parameters()}
        self.assertNotIn(id(actor.log_std), parameter_ids)
        self.assertEqual(actor.log_std.numel(), actor.action_dim)

        local_states = torch.zeros((3, 2, 4), dtype=torch.float32)
        action = torch.full((3, actor.action_dim), 0.75, dtype=torch.float32)
        loss = actor.evaluate_action(local_states, action).mean()
        loss.backward()
        self.assertIsNotNone(actor.log_std.grad)
        first_grad = actor.log_std.grad.detach().clone()

        actor.zero_grad()
        self.assertIsNotNone(actor.log_std.grad)
        self.assertTrue(torch.allclose(actor.log_std.grad, first_grad))

        actor.sample_action(local_states[0], use_mean=False)
        self.assertFalse(actor.log_std.requires_grad)


if __name__ == "__main__":
    unittest.main()
