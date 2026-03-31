import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Fused_CPRO import PRCRL_main, _build_scene, _maybe_initialize_new_actor_from_checkpoint
from run_mimo_prcrl import build_python_config as build_prcrl_python_config
from run_mimo_prcrl import _resolve_new_policy_init_args as resolve_prcrl_new_policy_init_args


def _save_sldac_checkpoint(
    root_dir,
    example_name,
    run_tag,
    seed,
    pretrain_episode,
    actor_state_dict,
    actor_log_std,
    state_dim,
    action_dim,
    constraint_dim,
):
    checkpoint_dir = os.path.join(root_dir, str(example_name), str(run_tag), "seed_{0}".format(int(seed)))
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "episode_{0:04d}.pt".format(int(pretrain_episode)))
    checkpoint = {
        "algorithm": "SLDAC",
        "example_name": str(example_name),
        "run_tag": str(run_tag),
        "seed": int(seed),
        "shapes": {
            "state_dim": int(state_dim),
            "action_dim": int(action_dim),
            "constraint_dim": int(constraint_dim),
        },
        "model": {
            "actor_state_dict": actor_state_dict,
            "actor_log_std": actor_log_std,
        },
    }
    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


class NewPolicyInitCheckpointTest(unittest.TestCase):
    def test_new_actor_init_loads_mu_only_and_keeps_log_std(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("MIMO", 0, "cpu", 1)
        _, source_actor, _, _, _, _ = _build_scene("MIMO", 7, "cpu", 1)

        with torch.no_grad():
            for idx, para in enumerate(source_actor.net.parameters()):
                para.copy_(torch.full_like(para, 0.05 * float(idx + 1)))
        source_state_dict = {key: value.detach().clone() for key, value in source_actor.net.state_dict().items()}
        checkpoint_log_std = torch.full((action_dim,), 1.25, dtype=torch.float)
        original_log_std = actor.log_std.detach().clone()
        original_state_dict = {key: value.detach().clone() for key, value in actor.net.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_sldac_checkpoint(
                tmpdir,
                "MIMO",
                "b100_q10",
                7,
                50,
                source_state_dict,
                checkpoint_log_std,
                state_dim,
                action_dim,
                constraint_dim,
            )
            args = SimpleNamespace(
                new_policy_run_tag="b100_q10",
                new_policy_pretrain_episode=50,
                new_policy_seed=7,
                new_policy_checkpoint_root=tmpdir,
            )

            _maybe_initialize_new_actor_from_checkpoint(args, "MIMO", actor, state_dim, action_dim, constraint_dim)

        changed = False
        for key, value in actor.net.state_dict().items():
            self.assertTrue(torch.allclose(value, source_state_dict[key]))
            if not torch.allclose(original_state_dict[key], value):
                changed = True
        self.assertTrue(changed)
        self.assertTrue(torch.allclose(actor.log_std, original_log_std))
        self.assertFalse(torch.allclose(actor.log_std, checkpoint_log_std))

    def test_new_actor_init_missing_checkpoint_raises(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("MIMO", 0, "cpu", 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                new_policy_run_tag="b100_q10",
                new_policy_pretrain_episode=50,
                new_policy_seed=7,
                new_policy_checkpoint_root=tmpdir,
            )
            with self.assertRaises(FileNotFoundError):
                _maybe_initialize_new_actor_from_checkpoint(args, "MIMO", actor, state_dim, action_dim, constraint_dim)

    def test_prcrl_run_exposes_new_policy_init_config_fields(self):
        config = build_prcrl_python_config()
        self.assertIn("new_policy_init", config)
        self.assertIn("new_policy_seed", config)
        self.assertIn("new_policy_pretrain_episode", config)
        self.assertIn("new_policy_checkpoint_root", config)

    def test_prcrl_resolve_new_policy_init_args_parses_single_bq_pair(self):
        args = SimpleNamespace(
            new_policy_init=(100, 10),
            new_policy_seed=7,
            new_policy_pretrain_episode=50,
            new_policy_checkpoint_root="tmp/checkpoints",
        )
        args = resolve_prcrl_new_policy_init_args(args)
        self.assertEqual(args.new_policy_run_tag, "b100_q10")
        self.assertEqual(args.new_policy_seed, 7)
        self.assertEqual(args.new_policy_pretrain_episode, 50)
        self.assertEqual(args.new_policy_checkpoint_root, "tmp/checkpoints")

    def test_prcrl_main_calls_new_actor_init_before_old_policy_library(self):
        _, actor, state_dim, action_dim, constraint_dim, constr_lim = _build_scene("MIMO", 0, "cpu", 1)
        fake_env = SimpleNamespace(reset=lambda: None)
        args = SimpleNamespace(
            seed=0,
            device="cpu",
            T=1,
            grad_T=1,
            num_new_data=1,
            update_time_per_episode=1,
            MAX_STEPS=2,
            alpha_pow=0.5,
            beta_actor_pow=0.6,
            beta_rho_pow=0.7,
            tau_reward=1.0,
            tau_cost=1.0,
            window=8,
            new_policy_run_tag="b100_q10",
            new_policy_pretrain_episode=50,
            new_policy_seed=7,
            new_policy_checkpoint_root="tmp/checkpoints",
            old_policy_run_tags="",
        )
        call_order = []

        def _fake_new_init(*_args, **_kwargs):
            call_order.append("init")
            return actor

        def _fake_old_policy_library(*_args, **_kwargs):
            call_order.append("old")
            raise RuntimeError("stop-after-init")

        with patch("Fused_CPRO._build_scene", return_value=(fake_env, actor, state_dim, action_dim, constraint_dim, constr_lim)):
            with patch("Fused_CPRO._maybe_initialize_new_actor_from_checkpoint", side_effect=_fake_new_init):
                with patch("Fused_CPRO._build_old_policy_library", side_effect=_fake_old_policy_library):
                    with self.assertRaisesRegex(RuntimeError, "stop-after-init"):
                        PRCRL_main(args, "MIMO")

        self.assertEqual(call_order, ["init", "old"])


if __name__ == "__main__":
    unittest.main()
