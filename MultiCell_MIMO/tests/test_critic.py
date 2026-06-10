import unittest

import torch


class CriticTest(unittest.TestCase):
    def test_dynamic_heads_and_target_modes(self):
        from MultiCell_MIMO.critic import MultiHeadDifferentialCritic

        critic = MultiHeadDifferentialCritic(
            state_dim=6,
            action_dim=3,
            constraint_dim=4,
            hidden_dims=(16,),
            device="cpu",
        )
        state = torch.randn(5, 6)
        action = torch.rand(5, 3)
        next_state = torch.randn(5, 6)
        next_action = torch.rand(5, 3)
        costs = torch.randn(5, 5)
        func_value = torch.zeros(5)

        source_target = critic.compute_td_target(
            costs,
            next_state,
            next_action,
            func_value,
            critic_target_mode="source_compatible",
        )
        strict_target = critic.compute_td_target(
            costs,
            next_state,
            next_action,
            func_value,
            critic_target_mode="tex_strict",
        )

        self.assertEqual(len(critic.heads), 5)
        self.assertEqual(source_target.shape, (5, 5))
        self.assertEqual(strict_target.shape, (5, 5))
        self.assertTrue(torch.isfinite(source_target).all().item())
        self.assertTrue(torch.isfinite(strict_target).all().item())

    def test_update_returns_finite_losses(self):
        from MultiCell_MIMO.critic import MultiHeadDifferentialCritic

        critic = MultiHeadDifferentialCritic(
            state_dim=4,
            action_dim=2,
            constraint_dim=1,
            hidden_dims=(8,),
            device="cpu",
        )
        losses = critic.update(
            state=torch.randn(6, 4),
            action=torch.rand(6, 2),
            costs=torch.randn(6, 2),
            next_state=torch.randn(6, 4),
            next_action=torch.rand(6, 2),
            func_value=torch.zeros(2),
            eta=0.1,
            gamma=0.5,
            critic_target_mode="source_compatible",
        )

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(value >= 0.0 for value in losses))


if __name__ == "__main__":
    unittest.main()
