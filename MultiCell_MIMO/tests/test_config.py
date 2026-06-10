import unittest


class ConfigTest(unittest.TestCase):
    def test_default_config_validates_required_modes(self):
        from MultiCell_MIMO.config import build_default_config, validate_config

        config = build_default_config()
        validate_config(config)

        self.assertEqual(config["critic_backend"], "centralized")
        self.assertEqual(config["critic_target_mode"], "source_compatible")
        self.assertEqual(config["actor_parameterization"], "shared")
        self.assertEqual(config["log_std_mode"], "joint")
        self.assertEqual(config["log_std_min"], -5.0)
        self.assertEqual(config["log_std_max"], 2.0)
        self.assertLess(config["log_std_min"], config["log_std_max"])
        self.assertEqual(config["run_id"], "")
        self.assertEqual(config["allow_overwrite"], 0)

    def test_invalid_enum_values_raise(self):
        from MultiCell_MIMO.config import build_default_config, validate_config

        for field_name in ("critic_backend", "critic_target_mode", "actor_parameterization", "log_std_mode"):
            config = build_default_config()
            config[field_name] = "invalid"
            with self.assertRaises(ValueError):
                validate_config(config)

        config = build_default_config()
        config["critic_target_mode"] = "tex_strict"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_cli_merge_keeps_protected_python_defaults(self):
        from MultiCell_MIMO.config import build_default_config, merge_cli_config

        config = build_default_config()
        merged, ignored = merge_cli_config(
            config,
            {"episode": 999, "device": "cuda"},
            protected_fields=("episode",),
        )

        self.assertEqual(merged["episode"], config["episode"])
        self.assertEqual(merged["device"], "cuda")
        self.assertIn("episode", ignored)


if __name__ == "__main__":
    unittest.main()
