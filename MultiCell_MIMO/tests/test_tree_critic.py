import unittest

import torch


class TreeCriticTest(unittest.TestCase):
    def test_tree_critic_values_and_source_compatible_target_mode_are_finite(self):
        from MultiCell_MIMO.tree_critic import TreeMessageDifferentialCritic

        critic = TreeMessageDifferentialCritic(
            local_state_dim=7,
            cell_count=3,
            cell_action_dim=2,
            constraint_dim=2,
            message_dim=8,
            hidden_dims=(16,),
            device="cpu",
        )
        local_state = torch.randn(4, 3, 7)
        action = torch.rand(4, 6)
        costs = torch.randn(4, 3)
        func_value = torch.zeros(3)

        online = critic.online_value(local_state, action)
        target_value = critic.critic_value(local_state, action, use_target=True)
        source_target = critic.compute_td_target(
            costs,
            local_state,
            action,
            func_value,
            critic_target_mode="source_compatible",
        )

        self.assertEqual(online.shape, (4, 3))
        self.assertEqual(target_value.shape, (4, 3))
        self.assertEqual(source_target.shape, (4, 3))
        self.assertTrue(torch.isfinite(online).all().item())
        with self.assertRaises(ValueError):
            critic.compute_td_target(
                costs,
                local_state,
                action,
                func_value,
                critic_target_mode="tex_strict",
            )

    def test_tree_critic_updates_averaged_encoder_and_heads(self):
        from MultiCell_MIMO.tree_critic import TreeMessageDifferentialCritic

        critic = TreeMessageDifferentialCritic(
            local_state_dim=5,
            cell_count=2,
            cell_action_dim=2,
            constraint_dim=1,
            message_dim=6,
            hidden_dims=(8,),
            device="cpu",
        )
        before = critic.flatten_target_parameters().detach().clone()
        losses = critic.update(
            local_state=torch.randn(5, 2, 5),
            action=torch.rand(5, 4),
            costs=torch.randn(5, 2),
            next_local_state=torch.randn(5, 2, 5),
            next_action=torch.rand(5, 4),
            func_value=torch.zeros(2),
            eta=0.1,
            gamma=0.5,
            critic_target_mode="source_compatible",
        )
        after = critic.flatten_target_parameters().detach().clone()

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(value >= 0.0 for value in losses))
        self.assertFalse(torch.allclose(before, after))


if __name__ == "__main__":
    unittest.main()
