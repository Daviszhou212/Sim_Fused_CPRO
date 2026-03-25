import argparse
import unittest

from seed_utils import apply_python_config_priority, format_ignored_cli_overrides, resolve_experiment_seeds


def build_python_config():
    return {
        "seed": 1,
        "seeds": "1,2,3,4",
        "episode": 60,
        "device": "cpu",
        "old_policy_pretrain_episode": 40,
        "old_policy_checkpoint_root": "checkpoints/SLDAC",
    }


PROTECTED_CLI_FIELDS = tuple(build_python_config().keys())


class CliConfigPriorityTest(unittest.TestCase):
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
            old_policy_pretrain_episode=12,
            old_policy_checkpoint_root="tmp/checkpoints",
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
                "--old-policy-pretrain-episode",
                "12",
                "--old-policy-checkpoint-root",
                "tmp/checkpoints",
            ],
        )
        self.assertEqual(args.seed, 1)
        self.assertEqual(args.episode, 60)
        self.assertEqual(args.old_policy_pretrain_episode, 40)
        self.assertEqual(args.old_policy_checkpoint_root, "checkpoints/SLDAC")
        self.assertEqual(
            ignored_options,
            [
                "--seed",
                "--episode",
                "--old-policy-pretrain-episode",
                "--old-policy-checkpoint-root",
            ],
        )
        self.assertIn("--old-policy-pretrain-episode", format_ignored_cli_overrides(ignored_options))

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


if __name__ == "__main__":
    unittest.main()
