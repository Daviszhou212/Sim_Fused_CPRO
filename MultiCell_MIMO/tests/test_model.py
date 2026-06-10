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


if __name__ == "__main__":
    unittest.main()
