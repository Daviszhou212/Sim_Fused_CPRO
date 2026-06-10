import importlib
import os
import sys
import unittest

import torch


THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CLQR_DIR = os.path.abspath(os.path.join(THIS_DIR, "..", "CLQR"))
for path in (THIS_DIR, CLQR_DIR):
    while path in sys.path:
        sys.path.remove(path)
sys.path.insert(0, CLQR_DIR)
sys.path.insert(0, THIS_DIR)

from seed_utils import resolve_torch_device  # noqa: E402
from model import (  # noqa: E402
    LEGACY_ACTOR_DISTRIBUTION,
    MIMO_POWER_MAX,
    SQUASHED_ACTOR_DISTRIBUTION,
    build_gaussian_policy,
    get_action_transform_metadata,
    normalize_actor_distribution,
)


GAUSSIAN_ENTRY_MODULES = (
    "run_mimo_sldac",
    "run_mimo_sldac_pathwise",
    "run_mimo_fused_cpro",
    "run_mimo_fused_cpro_cosrho",
    "run_mimo_fused_cpro_rho_new",
    "run_mimo_hrl",
    "run_mimo_prcrl",
)

GPU_AWARE_ENTRY_MODULES = GAUSSIAN_ENTRY_MODULES + (
    "run_mimo_acpo",
    "run_mimo_dk",
)


class ActorDistributionConfigTest(unittest.TestCase):
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

    def test_gaussian_entry_defaults_are_squashed(self):
        for module_name in GAUSSIAN_ENTRY_MODULES:
            with self.subTest(module=module_name):
                module = importlib.import_module(module_name)
                config = module.build_python_config()
                self.assertEqual(config["actor_distribution"], "squashed")
                self.assertIn("actor_distribution", module.PROTECTED_CLI_FIELDS)

    def test_dk_entries_do_not_expose_actor_distribution(self):
        mimo_dk = importlib.import_module("run_mimo_dk")
        self.assertNotIn("actor_distribution", mimo_dk.build_python_config())
        self.assertNotIn("actor_distribution", mimo_dk.PROTECTED_CLI_FIELDS)

        sys.path.insert(0, CLQR_DIR)
        try:
            clqr_dk = importlib.import_module("run_clqr_dk")
        finally:
            sys.path.pop(0)
            for module_name in (
                "run_clqr_dk",
                "Fused_CPRO",
                "environment",
                "artifact_paths",
                "seed_utils",
            ):
                sys.modules.pop(module_name, None)
        self.assertNotIn("actor_distribution", clqr_dk.build_python_config())
        self.assertNotIn("actor_distribution", clqr_dk.PROTECTED_CLI_FIELDS)

    def test_factory_selects_legacy_or_squashed_distribution(self):
        squashed = build_gaussian_policy("MIMO", 7, 5, "cpu", 4, actor_distribution="squashed")
        legacy = build_gaussian_policy("MIMO", 7, 5, "cpu", 4, actor_distribution="legacy")

        self.assertEqual(squashed.actor_distribution, SQUASHED_ACTOR_DISTRIBUTION)
        self.assertEqual(legacy.actor_distribution, LEGACY_ACTOR_DISTRIBUTION)
        self.assertEqual(normalize_actor_distribution("squashed"), SQUASHED_ACTOR_DISTRIBUTION)
        self.assertEqual(normalize_actor_distribution("legacy"), LEGACY_ACTOR_DISTRIBUTION)

        state_torch = torch.randn(3, 7, dtype=torch.float)
        action_torch = torch.full((3, 5), 0.5, dtype=torch.float)
        squashed_log_prob = squashed.evaluate_action(state_torch, action_torch)
        legacy_log_prob = legacy.evaluate_action(state_torch, action_torch)

        self.assertTrue(torch.isfinite(legacy_log_prob).all().item())
        self.assertFalse(torch.allclose(squashed_log_prob, legacy_log_prob))

    def test_legacy_metadata_matches_old_mimo_bounded_mean(self):
        metadata = get_action_transform_metadata("legacy")

        self.assertEqual(metadata["mimo_power"], ["legacy_bounded_mean_gaussian", 0.0, MIMO_POWER_MAX])
        self.assertEqual(metadata["mimo_reg"], ["legacy_bounded_mean_gaussian", 0.0, MIMO_POWER_MAX])

    def test_clqr_factory_is_available_from_clqr_model(self):
        original_path = list(sys.path)
        sys.path = [CLQR_DIR] + [item for item in original_path if item != THIS_DIR and item != CLQR_DIR]
        sys.modules.pop("model", None)
        try:
            clqr_model = importlib.import_module("model")
            actor = clqr_model.build_gaussian_policy(
                "CLQR",
                15,
                4,
                "cpu",
                4,
                actor_distribution="legacy",
            )
            self.assertEqual(actor.actor_distribution, clqr_model.LEGACY_ACTOR_DISTRIBUTION)

            mimo_actor = clqr_model.build_gaussian_policy(
                "MIMO",
                7,
                5,
                "cpu",
                4,
                actor_distribution="legacy",
            )
            state_torch = torch.randn(3, 7, dtype=torch.float)
            expected = clqr_model.MIMO_POWER_MAX * torch.sigmoid(mimo_actor.net(state_torch))
            actual = mimo_actor.mean_action_tensor(state_torch)
            self.assertTrue(torch.allclose(actual, expected))

            clqr_metadata = clqr_model.get_action_transform_metadata("legacy")
            self.assertEqual(clqr_metadata["mimo_reg"], ["legacy_bounded_mean_gaussian", 0.0, clqr_model.MIMO_POWER_MAX])
        finally:
            sys.modules.pop("model", None)
            sys.path = original_path


if __name__ == "__main__":
    unittest.main()
