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
        self.assertEqual(config["nt"], 8)
        self.assertEqual(config["users_per_cell"], 4)
        self.assertEqual(config["action_interface"], "snr_db")
        self.assertEqual(config["log_std_min"], -5.0)
        self.assertEqual(config["log_std_max"], 2.0)
        self.assertLess(config["log_std_min"], config["log_std_max"])
        self.assertEqual(config["centralized_critic_output_scale"], "auto")
        self.assertEqual(config["run_id"], "")
        self.assertEqual(config["allow_overwrite"], 0)
        self.assertEqual(config["log_interval_episodes"], 10)

    def test_default_sample_scale_matches_sldac_code_mimo1(self):
        from MultiCell_MIMO.config import build_default_config

        config = build_default_config()

        self.assertEqual(config["nt"], 8)
        self.assertEqual(config["users_per_cell"], 4)
        self.assertEqual(config["episode"], 60)
        self.assertEqual(config["update_time_per_episode"], 10)
        self.assertEqual(config["t_horizon"], 500)
        self.assertEqual(config["grad_batch_size"], 500)
        self.assertEqual(config["num_new_data"], 100)
        self.assertEqual(config["q_update_time"], 1)
        self.assertEqual(config["window"], 10000)

    def test_invalid_enum_values_raise(self):
        from MultiCell_MIMO.config import build_default_config, validate_config

        for field_name in (
            "critic_backend",
            "critic_target_mode",
            "actor_parameterization",
            "log_std_mode",
            "action_interface",
        ):
            config = build_default_config()
            config[field_name] = "invalid"
            with self.assertRaises(ValueError):
                validate_config(config)

        config = build_default_config()
        config["critic_target_mode"] = "tex_strict"
        with self.assertRaises(ValueError):
            validate_config(config)

    def test_centralized_critic_output_scale_accepts_auto_or_positive_numeric(self):
        from MultiCell_MIMO.config import build_default_config, validate_config

        config = build_default_config()
        config["centralized_critic_output_scale"] = 25.0
        self.assertEqual(validate_config(config)["centralized_critic_output_scale"], 25.0)

        config = build_default_config()
        config["centralized_critic_output_scale"] = 0.0
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
