import copy


CRITIC_TARGET_MODES = ("source_compatible", "tex_strict")
ACTOR_PARAMETERIZATIONS = ("shared", "per_cell")
LOG_STD_MODES = ("shared_cell", "joint")
CSSCA_SOLVERS = ("lagrangian_dual", "cvx")


def build_default_config():
    return {
        # 场景规模：默认使用很小的 smoke 规模，正式实验可在入口文件顶部改大。
        "seed": 0,
        "device": "cpu",
        "nt": 4,
        "cell_count": 3,
        "users_per_cell": 2,
        "constraint_limit": 1.2,
        "arrival_upper": 2.0,
        "queue_max": 5.0,
        "power_max": 2.5,
        # 训练步数：保持顶部配置优先，避免 CLI 意外覆盖正式实验设置。
        "episode": 2,
        "update_time_per_episode": 2,
        "t_horizon": 8,
        "grad_batch_size": 8,
        "num_new_data": 4,
        "q_update_time": 1,
        "window": 1000,
        # SLDAC 时间尺度超参数；gamma 是 critic averaging step，不是折扣因子。
        "alpha_pow": 0.6,
        "beta_pow": 0.7,
        "eta_pow": 0.01,
        "gamma_pow_reward": 0.3,
        "gamma_pow_cost": 0.3,
        "tau_objective": 1.0,
        "tau_constraint": 1.0,
        # 模式开关：显式区分源码兼容 TD target 与 tex strict target。
        "critic_target_mode": "source_compatible",
        "actor_parameterization": "shared",
        "log_std_mode": "shared_cell",
        "cssca_solver": "lagrangian_dual",
        "hidden_dims": (64, 64),
        "critic_hidden_dims": (64, 64),
        # 输出隔离：正式入口只能写入 MultiCell_MIMO 内部路径。
        "output_root": "MultiCell_MIMO/outputs",
        "checkpoint_root": "MultiCell_MIMO/checkpoints",
        "save_final_checkpoint": 1,
        "checkpoint_interval_episodes": 10,
        "run_tag": "multicell_sldac",
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
    if config["critic_target_mode"] not in CRITIC_TARGET_MODES:
        raise ValueError("unsupported critic_target_mode: {0}".format(config["critic_target_mode"]))
    if config["actor_parameterization"] not in ACTOR_PARAMETERIZATIONS:
        raise ValueError("unsupported actor_parameterization: {0}".format(config["actor_parameterization"]))
    if config["log_std_mode"] not in LOG_STD_MODES:
        raise ValueError("unsupported log_std_mode: {0}".format(config["log_std_mode"]))
    if config["cssca_solver"] not in CSSCA_SOLVERS:
        raise ValueError("unsupported cssca_solver: {0}".format(config["cssca_solver"]))

    for key in ("nt", "cell_count", "users_per_cell", "episode", "t_horizon", "grad_batch_size", "num_new_data"):
        if int(config[key]) <= 0:
            raise ValueError("{0} must be positive".format(key))
    for key in ("constraint_limit", "arrival_upper", "queue_max", "power_max"):
        if float(config[key]) <= 0.0:
            raise ValueError("{0} must be positive".format(key))

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
