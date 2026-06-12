import unittest

import torch


class CriticTest(unittest.TestCase):
    def test_dynamic_heads_and_source_compatible_target_mode(self):
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

        self.assertEqual(len(critic.heads), 5)
        self.assertEqual(source_target.shape, (5, 5))
        self.assertTrue(torch.isfinite(source_target).all().item())
        with self.assertRaises(ValueError):
            critic.compute_td_target(
                costs,
                next_state,
                next_action,
                func_value,
                critic_target_mode="tex_strict",
            )

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

    def test_legacy_critic_matches_sldac_code_optimizer_and_bounds(self):
        from MultiCell_MIMO.critic import LegacyMultiHeadDifferentialCritic

        critic = LegacyMultiHeadDifferentialCritic(
            state_dim=4,
            action_dim=2,
            constraint_dim=1,
            q_update_time=4,
            device="cpu",
        )
        self.assertAlmostEqual(critic.optimizers[0].param_groups[0]["lr"], 0.05)

        value = critic.critic_value(torch.randn(5, 4), torch.randn(5, 2), use_target=True)
        self.assertEqual(value.shape, (5, 2))
        self.assertLessEqual(float(value.max()), 10.0)
        self.assertGreaterEqual(float(value.min()), -10.0)

        losses = critic.update(
            state=torch.randn(5, 4),
            action=torch.randn(5, 2),
            costs=torch.randn(5, 2),
            next_state=torch.randn(5, 4),
            next_action=torch.randn(5, 2),
            func_value=torch.zeros(2),
            eta=0.7,
            gamma_reward=0.3,
            gamma_cost=0.2,
            critic_target_mode="source_compatible",
        )

        self.assertEqual(len(losses), 2)
        self.assertAlmostEqual(critic.optimizers[0].param_groups[0]["lr"], 0.05)
        self.assertTrue(all(value >= 0.0 for value in losses))

    def test_ctde_critic_auto_output_scale_matches_total_power_bound(self):
        from MultiCell_MIMO.config import build_default_config, validate_config
        from MultiCell_MIMO.environment import MultiCellMIMOEnv
        from MultiCell_MIMO.sldac import _build_critic

        config = build_default_config()
        config = validate_config(config)
        env = MultiCellMIMOEnv(
            seed=config["seed"],
            nt=config["nt"],
            cell_count=config["cell_count"],
            users_per_cell=config["users_per_cell"],
            arrival_upper=config["arrival_upper"],
            queue_max=config["queue_max"],
            action_interface=config["action_interface"],
        )
        critic = _build_critic(config, env, device="cpu")

        self.assertEqual(critic.output_scale, 30.0)
        value = critic.critic_value(torch.randn(5, env.state_dim), torch.randn(5, env.action_dim), use_target=True)
        self.assertLessEqual(float(value.max()), 30.0)
        self.assertGreaterEqual(float(value.min()), -30.0)

    def test_ctde_critic_auto_output_scale_uses_configured_user_count(self):
        from MultiCell_MIMO.config import build_default_config, validate_config
        from MultiCell_MIMO.environment import MultiCellMIMOEnv
        from MultiCell_MIMO.sldac import _build_critic

        config = build_default_config()
        config.update({"cell_count": 2, "users_per_cell": 3, "power_max": 2.5})
        config = validate_config(config)
        env = MultiCellMIMOEnv(
            seed=config["seed"],
            nt=config["nt"],
            cell_count=config["cell_count"],
            users_per_cell=config["users_per_cell"],
            arrival_upper=config["arrival_upper"],
            queue_max=config["queue_max"],
            action_interface=config["action_interface"],
        )
        critic = _build_critic(config, env, device="cpu")

        self.assertEqual(critic.output_scale, 15.0)


if __name__ == "__main__":
    unittest.main()
