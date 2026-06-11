import argparse
import importlib
import os
import sys
import unittest

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

from seed_utils import (
    apply_python_config_priority,
    build_mat_metadata_from_args,
    format_ignored_cli_overrides,
    resolve_torch_device,
    resolve_experiment_seeds,
)

GPU_AWARE_ENTRY_MODULES = (
    "run_clqr_sldac",
    "run_clqr_sldac_pathwise",
    "run_clqr_fused_cpro",
    "run_clqr_fused_cpro_cosrho",
    "run_clqr_fused_cpro_rho_new",
    "run_clqr_hrl",
    "run_clqr_prcrl",
    "run_clqr_acpo",
    "run_clqr_dk",
)


def build_python_config():
    return {
        "seed": 1,
        "seeds": "1,2,3,4",
        "episode": 60,
        "device": "cpu",
        "xi0": 0.5,
        "rho_min_new_actor": 0.2,
        "rho_min_old_policy": 1e-4,
        "old_policy_pretrain_episode": 40,
        "old_policy_checkpoint_root": "checkpoints/SLDAC",
        "new_policy_init": (100, 10),
        "new_policy_seed": 7,
        "new_policy_pretrain_episode": 35,
        "new_policy_checkpoint_root": "checkpoints/SLDAC/new_actor",
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


class CliConfigPriorityTest(unittest.TestCase):
    def test_resolve_torch_device_prefers_cuda_in_auto_mode(self):
        self.assertEqual(resolve_torch_device(None, cuda_is_available=lambda: True), "cuda")
        self.assertEqual(resolve_torch_device("auto", cuda_is_available=lambda: True), "cuda")
        self.assertEqual(resolve_torch_device("gpu", cuda_is_available=lambda: True), "cuda")
        self.assertEqual(resolve_torch_device("auto", cuda_is_available=lambda: False), "cpu")
        self.assertEqual(resolve_torch_device("cuda", cuda_is_available=lambda: False), "cpu")
        self.assertEqual(resolve_torch_device("cpu", cuda_is_available=lambda: True), "cpu")

    def test_entry_defaults_use_auto_device(self):
        for module_name in GPU_AWARE_ENTRY_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                self.assertEqual(module.build_python_config()["device"], "auto")
                self.assertIn("device", module.PROTECTED_CLI_FIELDS)

    def test_python_config_wins_without_cli_args(self):
        args, ignored_options = apply_python_config_priority(
            argparse.Namespace(),
            build_python_config(),
            PROTECTED_CLI_FIELDS,
            argv=[],
        )
        self.assertEqual(vars(args), build_python_config())
        self.assertEqual(ignored_options, [])

    def test_conflicting_cli_args_are_ignored_with_warning(self):
        cli_args = argparse.Namespace(
            seed=99,
            episode=5,
            xi0=0.9,
            rho_min_new_actor=0.35,
            rho_min_old_policy=0.02,
            old_policy_pretrain_episode=12,
            old_policy_checkpoint_root="tmp/checkpoints",
            new_policy_init="b500:q10",
            new_policy_seed=9,
            new_policy_pretrain_episode=22,
            new_policy_checkpoint_root="tmp/new-checkpoints",
        )
        args, ignored_options = apply_python_config_priority(
            cli_args,
            build_python_config(),
            PROTECTED_CLI_FIELDS,
            argv=[
                "--seed",
                "99",
                "--episode",
                "5",
                "--xi0",
                "0.9",
                "--rho-min-new-actor",
                "0.35",
                "--rho-min-old-policy",
                "0.02",
                "--old-policy-pretrain-episode",
                "12",
                "--old-policy-checkpoint-root",
                "tmp/checkpoints",
                "--new-policy-init",
                "b500:q10",
                "--new-policy-seed",
                "9",
                "--new-policy-pretrain-episode",
                "22",
                "--new-policy-checkpoint-root",
                "tmp/new-checkpoints",
            ],
        )
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.episode, 60)
        self.assertEqual(args.xi0, 0.5)
        self.assertEqual(args.rho_min_new_actor, 0.2)
        self.assertEqual(args.rho_min_old_policy, 1e-4)
        self.assertEqual(args.old_policy_pretrain_episode, 40)
        self.assertEqual(args.old_policy_checkpoint_root, "checkpoints/SLDAC")
        self.assertEqual(args.new_policy_init, (100, 10))
        self.assertEqual(args.new_policy_seed, 7)
        self.assertEqual(args.new_policy_pretrain_episode, 35)
        self.assertEqual(args.new_policy_checkpoint_root, "checkpoints/SLDAC/new_actor")
        self.assertEqual(
            ignored_options,
            [
                "--seed",
                "--episode",
                "--xi0",
                "--rho-min-new-actor",
                "--rho-min-old-policy",
                "--old-policy-pretrain-episode",
                "--old-policy-checkpoint-root",
                "--new-policy-init",
                "--new-policy-seed",
                "--new-policy-pretrain-episode",
                "--new-policy-checkpoint-root",
            ],
        )
        ignored_message = format_ignored_cli_overrides(ignored_options)
        self.assertIn("--old-policy-pretrain-episode", ignored_message)
        self.assertIn("--new-policy-pretrain-episode", ignored_message)

    def test_seed_and_seeds_cli_do_not_change_experiment_seed_set(self):
        cli_args = argparse.Namespace(seed=99, seeds="9,10")
        args, ignored_options = apply_python_config_priority(
            cli_args,
            build_python_config(),
            PROTECTED_CLI_FIELDS,
            argv=["--seed", "99", "--seeds", "9,10"],
        )
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.seeds, "1,2,3,4")
        self.assertEqual(ignored_options, ["--seed", "--seeds"])
        self.assertEqual(resolve_experiment_seeds(args, 0), [1, 2, 3, 4])

    def test_build_mat_metadata_contains_algorithm_params(self):
        args = argparse.Namespace(
            seed=3,
            xi0=0.5,
            load_new_actor=False,
            new_policy_init=(100, 10),
            old_policies=None,
        )
        metadata = build_mat_metadata_from_args(args, "Fused_CPRO", "b100_q1", 0)

        self.assertIn("algorithm_params", metadata)
        self.assertEqual(int(metadata["seed"][0][0]), 3)
        self.assertEqual(str(metadata["algorithm"][0]), "Fused_CPRO")
        self.assertEqual(str(metadata["run_tag"][0]), "b100_q1")

        params = metadata["algorithm_params"]
        self.assertEqual(int(params["seed"][0][0]), 3)
        self.assertAlmostEqual(float(params["xi0"][0][0]), 0.5)
        self.assertFalse(bool(params["load_new_actor"][0][0]))
        self.assertEqual(params["new_policy_init"].tolist(), [[100, 10]])
        self.assertEqual(str(params["old_policies"][0]), "None")


if __name__ == "__main__":
    unittest.main()
