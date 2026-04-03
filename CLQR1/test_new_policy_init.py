import os
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

from Fused_CPRO import (
    HRL_main,
    PRCRL_main,
    _build_scene,
    _load_old_policy_from_checkpoint,
    _maybe_initialize_new_actor_from_checkpoint,
)
from run_clqr_fused_cpro import build_python_config as build_fused_cpro_python_config
from run_clqr_hrl import build_python_config as build_hrl_python_config
from run_clqr_hrl import _resolve_new_policy_init_args as resolve_hrl_new_policy_init_args
from run_clqr_prcrl import build_python_config as build_prcrl_python_config
from run_clqr_prcrl import _resolve_new_policy_init_args as resolve_prcrl_new_policy_init_args


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
    def test_fused_cpro_run_exposes_log_std_switches(self):
        config = build_fused_cpro_python_config()
        self.assertIn("load_new_actor_log_std", config)
        self.assertIn("load_old_policy_log_std", config)
        self.assertFalse(config["load_new_actor_log_std"])
        self.assertFalse(config["load_old_policy_log_std"])

    def test_new_actor_init_loads_mu_only_and_keeps_log_std(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 1)
        _, source_actor, _, _, _, _ = _build_scene("CLQR", 7, "cpu", 1)

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
                "CLQR",
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

            _maybe_initialize_new_actor_from_checkpoint(args, "CLQR", actor, state_dim, action_dim, constraint_dim)

        changed = False
        for key, value in actor.net.state_dict().items():
            self.assertTrue(torch.allclose(value, source_state_dict[key]))
            if not torch.allclose(original_state_dict[key], value):
                changed = True
        self.assertTrue(changed)
        self.assertTrue(torch.allclose(actor.log_std, original_log_std))
        self.assertFalse(torch.allclose(actor.log_std, checkpoint_log_std))

    def test_new_actor_init_loads_checkpoint_log_std_when_enabled(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 1)
        _, source_actor, _, _, _, _ = _build_scene("CLQR", 7, "cpu", 1)

        with torch.no_grad():
            for idx, para in enumerate(source_actor.net.parameters()):
                para.copy_(torch.full_like(para, 0.05 * float(idx + 1)))
        source_state_dict = {key: value.detach().clone() for key, value in source_actor.net.state_dict().items()}
        checkpoint_log_std = torch.full((action_dim,), 1.25, dtype=torch.float)

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_sldac_checkpoint(
                tmpdir,
                "CLQR",
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
                load_new_actor_log_std=True,
            )

            _maybe_initialize_new_actor_from_checkpoint(args, "CLQR", actor, state_dim, action_dim, constraint_dim)

        self.assertTrue(torch.allclose(actor.log_std, checkpoint_log_std))

    def test_new_actor_init_skips_missing_log_std_when_disabled(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 1)
        _, source_actor, _, _, _, _ = _build_scene("CLQR", 7, "cpu", 1)
        source_state_dict = {key: value.detach().clone() for key, value in source_actor.net.state_dict().items()}
        original_log_std = actor.log_std.detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "CLQR", "b100_q10", "seed_{0}".format(7))
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "episode_{0:04d}.pt".format(50))
            torch.save(
                {
                    "algorithm": "SLDAC",
                    "example_name": "CLQR",
                    "run_tag": "b100_q10",
                    "seed": 7,
                    "shapes": {
                        "state_dim": int(state_dim),
                        "action_dim": int(action_dim),
                        "constraint_dim": int(constraint_dim),
                    },
                    "model": {"actor_state_dict": source_state_dict},
                },
                checkpoint_path,
            )
            args = SimpleNamespace(
                new_policy_run_tag="b100_q10",
                new_policy_pretrain_episode=50,
                new_policy_seed=7,
                new_policy_checkpoint_root=tmpdir,
                load_new_actor_log_std=False,
            )

            _maybe_initialize_new_actor_from_checkpoint(args, "CLQR", actor, state_dim, action_dim, constraint_dim)

        self.assertTrue(torch.allclose(actor.log_std, original_log_std))

    def test_old_policy_checkpoint_loads_mu_only_and_keeps_default_log_std(self):
        _, source_actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 7, "cpu", 1)
        _, default_actor, _, _, _, _ = _build_scene("CLQR", 0, "cpu", 1)

        with torch.no_grad():
            for idx, para in enumerate(source_actor.net.parameters()):
                para.copy_(torch.full_like(para, 0.05 * float(idx + 1)))
        source_state_dict = {key: value.detach().clone() for key, value in source_actor.net.state_dict().items()}
        checkpoint_log_std = torch.full((action_dim,), 1.25, dtype=torch.float)
        default_log_std = default_actor.log_std.detach().clone()

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_sldac_checkpoint(
                tmpdir,
                "CLQR",
                "b100_q10",
                7,
                50,
                source_state_dict,
                checkpoint_log_std,
                state_dim,
                action_dim,
                constraint_dim,
            )
            args = SimpleNamespace()

            frozen_policy = _load_old_policy_from_checkpoint(
                args,
                "CLQR",
                "b100_q10",
                50,
                7,
                "cpu",
                1,
                state_dim,
                action_dim,
                constraint_dim,
                checkpoint_root=tmpdir,
            )

        for key, value in frozen_policy.actor.net.state_dict().items():
            self.assertTrue(torch.allclose(value, source_state_dict[key]))
        self.assertTrue(torch.allclose(frozen_policy.actor.log_std, default_log_std))
        self.assertFalse(torch.allclose(frozen_policy.actor.log_std, checkpoint_log_std))

    def test_old_policy_checkpoint_loads_checkpoint_log_std_when_enabled(self):
        _, source_actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 7, "cpu", 1)

        with torch.no_grad():
            for idx, para in enumerate(source_actor.net.parameters()):
                para.copy_(torch.full_like(para, 0.05 * float(idx + 1)))
        source_state_dict = {key: value.detach().clone() for key, value in source_actor.net.state_dict().items()}
        checkpoint_log_std = torch.full((action_dim,), 1.25, dtype=torch.float)

        with tempfile.TemporaryDirectory() as tmpdir:
            _save_sldac_checkpoint(
                tmpdir,
                "CLQR",
                "b100_q10",
                7,
                50,
                source_state_dict,
                checkpoint_log_std,
                state_dim,
                action_dim,
                constraint_dim,
            )
            args = SimpleNamespace(load_old_policy_log_std=True)

            frozen_policy = _load_old_policy_from_checkpoint(
                args,
                "CLQR",
                "b100_q10",
                50,
                7,
                "cpu",
                1,
                state_dim,
                action_dim,
                constraint_dim,
                checkpoint_root=tmpdir,
            )

        self.assertTrue(torch.allclose(frozen_policy.actor.log_std, checkpoint_log_std))

    def test_old_policy_checkpoint_missing_log_std_raises_when_enabled(self):
        _, source_actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 7, "cpu", 1)
        source_state_dict = {key: value.detach().clone() for key, value in source_actor.net.state_dict().items()}

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = os.path.join(tmpdir, "CLQR", "b100_q10", "seed_{0}".format(7))
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = os.path.join(checkpoint_dir, "episode_{0:04d}.pt".format(50))
            torch.save(
                {
                    "algorithm": "SLDAC",
                    "example_name": "CLQR",
                    "run_tag": "b100_q10",
                    "seed": 7,
                    "shapes": {
                        "state_dim": int(state_dim),
                        "action_dim": int(action_dim),
                        "constraint_dim": int(constraint_dim),
                    },
                    "model": {"actor_state_dict": source_state_dict},
                },
                checkpoint_path,
            )
            args = SimpleNamespace(load_old_policy_log_std=True)

            with self.assertRaisesRegex(KeyError, "actor_log_std"):
                _load_old_policy_from_checkpoint(
                    args,
                    "CLQR",
                    "b100_q10",
                    50,
                    7,
                    "cpu",
                    1,
                    state_dim,
                    action_dim,
                    constraint_dim,
                    checkpoint_root=tmpdir,
                )

    def test_new_actor_init_missing_checkpoint_raises(self):
        _, actor, state_dim, action_dim, constraint_dim, _ = _build_scene("CLQR", 0, "cpu", 1)
        with tempfile.TemporaryDirectory() as tmpdir:
            args = SimpleNamespace(
                new_policy_run_tag="b100_q10",
                new_policy_pretrain_episode=50,
                new_policy_seed=7,
                new_policy_checkpoint_root=tmpdir,
            )
            with self.assertRaises(FileNotFoundError):
                _maybe_initialize_new_actor_from_checkpoint(args, "CLQR", actor, state_dim, action_dim, constraint_dim)

    def test_prcrl_run_exposes_new_policy_init_config_fields(self):
        config = build_prcrl_python_config()
        self.assertIn("new_policy_init", config)
        self.assertIn("new_policy_seed", config)
        self.assertIn("new_policy_pretrain_episode", config)
        self.assertIn("new_policy_checkpoint_root", config)

    def test_prcrl_resolve_new_policy_init_args_parses_single_bq_pair(self):
        args = SimpleNamespace(
            load_new_actor=True,
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

    def test_prcrl_resolve_new_policy_init_args_skips_run_tag_when_load_disabled(self):
        args = SimpleNamespace(
            load_new_actor=False,
            new_policy_init=(100, 10),
            new_policy_seed=7,
            new_policy_pretrain_episode=50,
            new_policy_checkpoint_root="tmp/checkpoints",
        )
        args = resolve_prcrl_new_policy_init_args(args)
        self.assertEqual(args.new_policy_run_tag, "")

    def test_hrl_run_exposes_new_policy_init_config_fields(self):
        config = build_hrl_python_config()
        self.assertIn("new_policy_init", config)
        self.assertIn("new_policy_seed", config)
        self.assertIn("new_policy_pretrain_episode", config)
        self.assertIn("new_policy_checkpoint_root", config)
        self.assertIn("load_new_actor_log_std", config)
        self.assertIn("load_old_policy_log_std", config)

    def test_hrl_resolve_new_policy_init_args_parses_single_bq_pair(self):
        args = SimpleNamespace(
            load_new_actor=True,
            new_policy_init=(100, 10),
            new_policy_seed=7,
            new_policy_pretrain_episode=50,
            new_policy_checkpoint_root="tmp/checkpoints",
        )
        args = resolve_hrl_new_policy_init_args(args)
        self.assertEqual(args.new_policy_run_tag, "b100_q10")
        self.assertEqual(args.new_policy_seed, 7)
        self.assertEqual(args.new_policy_pretrain_episode, 50)
        self.assertEqual(args.new_policy_checkpoint_root, "tmp/checkpoints")

    def test_hrl_resolve_new_policy_init_args_skips_run_tag_when_load_disabled(self):
        args = SimpleNamespace(
            load_new_actor=False,
            new_policy_init=(100, 10),
            new_policy_seed=7,
            new_policy_pretrain_episode=50,
            new_policy_checkpoint_root="tmp/checkpoints",
        )
        args = resolve_hrl_new_policy_init_args(args)
        self.assertEqual(args.new_policy_run_tag, "")

    def test_prcrl_main_calls_new_actor_init_before_old_policy_library(self):
        _, actor, state_dim, action_dim, constraint_dim, constr_lim = _build_scene("CLQR", 0, "cpu", 1)
        fake_env = SimpleNamespace(reset=lambda: None)
        args = SimpleNamespace(
            seed=0,
            device="cpu",
            T=1,
            grad_T=1,
            num_new_data=1,
            update_time_per_episode=1,
            MAX_STEPS=2,
            alpha_pow=0.6,
            beta_pow=0.8,
            beta_actor_pow=0.8,
            beta_rho_pow=0.9,
            tau_reward=6.0,
            tau_cost=10.0,
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
                        PRCRL_main(args, "CLQR")

        self.assertEqual(call_order, ["init", "old"])

    def test_hrl_main_calls_new_actor_init_before_old_policy_library(self):
        _, actor, state_dim, action_dim, constraint_dim, constr_lim = _build_scene("CLQR", 0, "cpu", 1)
        fake_env = SimpleNamespace(reset=lambda: None)
        args = SimpleNamespace(
            seed=0,
            device="cpu",
            T=1,
            grad_T=1,
            num_new_data=1,
            update_time_per_episode=1,
            MAX_STEPS=2,
            alpha_pow=0.6,
            beta_pow=0.8,
            beta_actor_pow=0.8,
            beta_rho_pow=0.9,
            eta_pow=0.01,
            gamma_pow_reward=0.3,
            gamma_pow_cost=0.3,
            tau_reward=6.0,
            tau_cost=10.0,
            Q_update_time=1,
            window=8,
            load_new_actor=True,
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
                        HRL_main(args, "CLQR", return_aux=True)

        self.assertEqual(call_order, ["init", "old"])


if __name__ == "__main__":
    unittest.main()
