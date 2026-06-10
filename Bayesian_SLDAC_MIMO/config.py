from dataclasses import dataclass, replace


SLDAC_RUNS = {
    "b100_q10": ("Bayesian SLDAC, batchsize=100, q=10", 500, 500, 100, 10),
    "b100_q1": ("Bayesian SLDAC, batchsize=100, q=1", 500, 500, 100, 1),
    "b100_q5": ("Bayesian SLDAC, batchsize=100, q=5", 500, 500, 100, 5),
    "b500_q10": ("Bayesian SLDAC, source b500_q10 setting", 50, 100, 100, 10),
}


@dataclass
class RunConfig:
    example_name: str = "MIMO"
    run_tag: str = "b100_q1"
    run_label: str = "Bayesian SLDAC, batchsize=100, q=1"
    seed: int = 0
    device: str = "cpu"
    alpha_pow: float = 0.6
    beta_pow: float = 0.7
    eta_pow: float = 0.01
    gamma_pow_reward: float = 0.3
    gamma_pow_cost: float = 0.3
    tau_reward: float = 1.0
    tau_cost: float = 1.0
    T: int = 500
    grad_T: int = 500
    num_new_data: int = 100
    window: int = 10000
    episode: int = 60
    update_time_per_episode: int = 10
    num_update_time: int = 600
    Q_update_time: int = 1
    MAX_STEPS: int = 61000
    ensemble_size: int = 5
    beta_uncertainty: float = 0.0
    bootstrap_mask_prob: float = 1.0
    # critic Adam 基础学习率，实际 lr=critic_lr_base/sqrt(Q_update_time)。
    critic_lr_base: float = 0.01
    # Bayesian critic 独立随机流种子，避免额外 ensemble 扰动 actor 采样流。
    critic_seed: int = 10000
    # ensemble 初始化方式：shared 避免未训练 member 初始饱和分歧；independent 用于诊断对照。
    ensemble_init_mode: str = "shared"
    cssca_solver: str = "lagrangian"


def _with_derived(config):
    num_update_time = int(config.episode) * int(config.update_time_per_episode)
    max_steps = 2 * int(config.T) + num_update_time * int(config.num_new_data)
    return replace(config, num_update_time=num_update_time, MAX_STEPS=max_steps)


def make_run_config(run_tag, episode_override=None, overrides=None):
    if run_tag not in SLDAC_RUNS:
        raise ValueError("unknown run_tag: {0}".format(run_tag))
    run_label, t_horizon, grad_t, num_new_data, q_update_time = SLDAC_RUNS[run_tag]
    config = RunConfig(
        run_tag=str(run_tag),
        run_label=str(run_label),
        T=int(t_horizon),
        grad_T=int(grad_t),
        num_new_data=int(num_new_data),
        Q_update_time=int(q_update_time),
    )
    if episode_override is not None:
        config = replace(config, episode=int(episode_override))
    if overrides:
        valid = set(config.__dataclass_fields__.keys())
        unknown = sorted(set(overrides.keys()) - valid)
        if unknown:
            raise ValueError("unknown config override(s): {0}".format(", ".join(unknown)))
        config = replace(config, **overrides)
    return _with_derived(config)
