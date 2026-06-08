import unittest
import argparse

import numpy as np
import torch

from critic_opt import Critic
from environment import Environment_MultiCellMIMO_CTDE
from model import GaussianPolicy_MultiCellMIMO_CTDE
from SLDAC import SLDAC_main


class CtdeMimoTest(unittest.TestCase):
    def test_multicell_environment_shapes_and_step(self):
        env = Environment_MultiCellMIMO_CTDE(seed=0, Nt=4, cell_num=3, user_per_cell=2)
        state = env.reset()
        self.assertEqual(tuple(state.shape), (env.state_dim,))
        self.assertEqual(env.action_dim, 9)
        self.assertEqual(env.constraint_dim, 6)

        action = np.full(env.action_dim, 0.5, dtype=np.float64)
        next_state, reward, done, info = env.step(action)
        self.assertEqual(tuple(next_state.shape), (env.state_dim,))
        self.assertFalse(done)
        self.assertTrue(np.isfinite(reward))
        self.assertIn("cost_6", info)
        self.assertTrue(np.isfinite(info["cost"]))
        self.assertEqual(tuple(env.local_observation(1).shape), (env.local_state_dim,))

    def test_ctde_actor_global_and_local_actions(self):
        env = Environment_MultiCellMIMO_CTDE(seed=1, Nt=4, cell_num=3, user_per_cell=2)
        state = env.reset()
        actor = GaussianPolicy_MultiCellMIMO_CTDE(
            env.state_dim,
            env.action_dim,
            "cpu",
            4,
            cell_num=env.cell_num,
            user_per_cell=env.user_per_cell,
            Nt=env.Nt,
        )

        action = actor.sample_action(state, use_mean=True)
        self.assertEqual(tuple(action.shape), (env.action_dim,))
        self.assertTrue(np.isfinite(action).all())

        state_batch = torch.tensor(np.stack([state, state]), dtype=torch.float)
        mean_action = actor.mean_action_tensor(state_batch)
        self.assertEqual(tuple(mean_action.shape), (2, env.action_dim))

        local_action = actor.sample_cell_action(env.local_observation(2), cell_index=2, use_mean=True)
        self.assertEqual(tuple(local_action.shape), (env.cell_action_dim,))
        self.assertTrue(np.isfinite(local_action).all())

    def test_dynamic_mimo_critic_supports_ctde_constraint_count(self):
        env = Environment_MultiCellMIMO_CTDE(seed=2, Nt=4, cell_num=3, user_per_cell=2)
        batch_size = 5
        critic = Critic("MIMO_CTDE", batch_size, env.state_dim, env.action_dim, env.constraint_dim, 1, "cpu")
        state = np.random.randn(batch_size, env.state_dim)
        action = np.random.rand(batch_size, env.action_dim)
        costs = np.random.randn(batch_size, 1 + env.constraint_dim)
        next_state = np.random.randn(batch_size, env.state_dim)
        next_action = np.random.rand(batch_size, env.action_dim)

        critic.critic_update(
            np.zeros(1 + env.constraint_dim, dtype=np.float64),
            state,
            action,
            costs,
            next_state,
            next_action,
            eta=1.0,
            gamma_reward=0.0,
            gamma_cost=0.0,
        )
        q_hat = critic.critic_value(
            torch.tensor(state, dtype=torch.float),
            torch.tensor(action, dtype=torch.float),
        )
        self.assertEqual(tuple(q_hat.shape), (batch_size, 1 + env.constraint_dim))
        self.assertTrue(torch.isfinite(q_hat).all().item())
        self.assertTrue(hasattr(critic, "net6"))
        self.assertTrue(hasattr(critic, "target_net6"))

    def test_sldac_can_return_per_user_delay_history(self):
        args = argparse.Namespace(
            seed=0,
            device="cpu",
            T=2,
            grad_T=2,
            num_new_data=2,
            update_time_per_episode=1,
            num_update_time=1,
            MAX_STEPS=2 * 2 + (1 + 1) * 2,
            alpha_pow=0.6,
            beta_pow=0.7,
            eta_pow=0.01,
            gamma_pow_reward=0.3,
            gamma_pow_cost=0.3,
            tau_reward=1.0,
            tau_cost=1.0,
            Q_update_time=1,
            window=20,
            episode=1,
            Nt=2,
            num_cells=2,
            users_per_cell=1,
            constraint_limit=1.2,
            checkpoint_interval_episodes=9999,
            save_final_checkpoint=0,
            return_per_user_delay=1,
        )
        reward_history, delay_history, per_user_delay = SLDAC_main(args, "MIMO_CTDE")
        self.assertEqual(len(reward_history), 1)
        self.assertEqual(len(delay_history), 1)
        self.assertEqual(tuple(per_user_delay.shape), (1, 2))
        self.assertTrue(np.isfinite(per_user_delay).all())


if __name__ == "__main__":
    unittest.main()
