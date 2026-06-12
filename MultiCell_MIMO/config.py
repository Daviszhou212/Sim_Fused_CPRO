import copy


CRITIC_BACKENDS = ("centralized", "tree")
CRITIC_TARGET_MODES = ("source_compatible",)
ACTOR_PARAMETERIZATIONS = ("shared", "per_cell")
LOG_STD_MODES = ("shared_cell", "joint")
CSSCA_SOLVERS = ("lagrangian_dual", "cvx")
ACTION_INTERFACES = ("legacy_power", "snr_db")


def build_default_config():
    return {
        # 场景规模：默认对齐 SLDAC_code/MIMO1 的正式样本量；快速检查应在测试或临时 runner 中覆盖为 smoke 规模。
        "seed": 0,
        "device": "cpu",
        "nt": 4,
        "cell_count": 3,
        "users_per_cell": 2,
        "constraint_limit": 1.2,
        "arrival_upper": 2.0,
        "queue_max": 5.0,
        "power_max": 2.5,
        "action_interface": "snr_db",
        # 训练步数：与 SLDAC_code/MIMO1 默认入口保持一致；入口脚本主要通过本文件配置。
        "episode": 60,
        "update_time_per_episode": 10,
        "t_horizon": 500,
        "grad_batch_size": 500,
        "num_new_data": 100,
        "q_update_time": 1,
        "window": 10000,
        # SLDAC 时间尺度超参数；gamma 是 critic averaging step，不是折扣因子。
        "alpha_pow": 0.6,
        "beta_pow": 0.7,
        "eta_pow": 0.01,
        "gamma_pow_reward": 0.3,
        "gamma_pow_cost": 0.3,
        "tau_objective": 1.0,
        "tau_constraint": 1.0,
        # Critic TD target 固定使用 SLDAC 源码兼容口径：bootstrap 来自平滑 target critic。
        "critic_target_mode": "source_compatible",
        "critic_backend": "centralized",
        # CTDE centralized critic 输出尺度；auto 按小区数扩展原始 MIMO 的 10*tanh bound。
        "centralized_critic_output_scale": "auto",
        "actor_parameterization": "shared",
        "log_std_mode": "joint",
        "log_std_min": -5.0,
        "log_std_max": 2.0,
        "cssca_solver": "lagrangian_dual",
        "hidden_dims": (128, 128),
        "critic_hidden_dims": (64, 64),
        "tree_message_dim": 32,
        # 输出隔离：正式入口只能写入 MultiCell_MIMO 内部路径。
        "output_root": "MultiCell_MIMO/outputs",
        "checkpoint_root": "MultiCell_MIMO/checkpoints",
        # 进度与检查点：长仿真按 episode 间隔打印，并可按需保存最终 checkpoint。
        "save_final_checkpoint": 1,
        "checkpoint_interval_episodes": 10,
        "log_interval_episodes": 10,
        "run_tag": "multicell_sldac",
        "run_id": "",
        "allow_overwrite": 0,
    }


def _as_tuple(value):
    if isinstance(value, tuple):
        return value
    if isinstance(value, list):
        return tuple(value)
    if isinstance(value, str):
        return tuple(int(item.strip()) for item in value.split(",") if item.strip())
    return value


def validate_config(config):
    config.setdefault("critic_backend", "centralized")
    config.setdefault("tree_message_dim", 32)
    config.setdefault("q_update_time", 1)
    config.setdefault("run_id", "")
    config.setdefault("allow_overwrite", 0)
    config.setdefault("action_interface", "snr_db")
    config.setdefault("log_interval_episodes", 10)
    config.setdefault("centralized_critic_output_scale", "auto")
    if config["critic_backend"] not in CRITIC_BACKENDS:
        raise ValueError("unsupported critic_backend: {0}".format(config["critic_backend"]))
    if config["critic_target_mode"] not in CRITIC_TARGET_MODES:
        raise ValueError("unsupported critic_target_mode: {0}".format(config["critic_target_mode"]))
    if config["actor_parameterization"] not in ACTOR_PARAMETERIZATIONS:
        raise ValueError("unsupported actor_parameterization: {0}".format(config["actor_parameterization"]))
    if config["log_std_mode"] not in LOG_STD_MODES:
        raise ValueError("unsupported log_std_mode: {0}".format(config["log_std_mode"]))
    if config["cssca_solver"] not in CSSCA_SOLVERS:
        raise ValueError("unsupported cssca_solver: {0}".format(config["cssca_solver"]))
    if config["action_interface"] not in ACTION_INTERFACES:
        raise ValueError("unsupported action_interface: {0}".format(config["action_interface"]))

    for key in (
        "nt",
        "cell_count",
        "users_per_cell",
        "episode",
        "update_time_per_episode",
        "t_horizon",
        "grad_batch_size",
        "num_new_data",
        "q_update_time",
        "window",
        "log_interval_episodes",
    ):
        if int(config[key]) <= 0:
            raise ValueError("{0} must be positive".format(key))
    for key in ("constraint_limit", "arrival_upper", "queue_max", "power_max"):
        if float(config[key]) <= 0.0:
            raise ValueError("{0} must be positive".format(key))
    critic_scale = config["centralized_critic_output_scale"]
    if isinstance(critic_scale, str) and critic_scale.strip().lower() == "auto":
        config["centralized_critic_output_scale"] = "auto"
    else:
        config["centralized_critic_output_scale"] = float(critic_scale)
        if float(config["centralized_critic_output_scale"]) <= 0.0:
            raise ValueError("centralized_critic_output_scale must be positive or auto")
    if float(config["log_std_min"]) >= float(config["log_std_max"]):
        raise ValueError("log_std_min must be smaller than log_std_max")
    if int(config["tree_message_dim"]) <= 0:
        raise ValueError("tree_message_dim must be positive")
    if int(config["num_new_data"]) % int(config["q_update_time"]) != 0:
        raise ValueError("num_new_data must be divisible by q_update_time")
    if int(config["window"]) < int(config["num_new_data"]):
        raise ValueError("window must be at least num_new_data")

    config["hidden_dims"] = _as_tuple(config["hidden_dims"])
    config["critic_hidden_dims"] = _as_tuple(config["critic_hidden_dims"])
    return config


def merge_cli_config(default_config, cli_values, protected_fields=None):
    merged = copy.deepcopy(default_config)
    protected = set(protected_fields or ())
    ignored = []
    for key, value in dict(cli_values or {}).items():
        if value is None:
            continue
        if key in protected:
            ignored.append(key)
            continue
        merged[key] = value
    return validate_config(merged), ignored
