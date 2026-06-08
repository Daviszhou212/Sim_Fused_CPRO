"""从现有 Fused_CPRO 实现中提炼出的 DK 策略逻辑展示文件。

该文件只用于集中呈现 MIMO 与 CLQR 两类 DK 策略的核心决策逻辑，
不接入当前训练、评估、buffer、critic 或 DK_main 运行流程。
"""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

try:
    import torch
    import torch.nn.functional as F
except Exception:  # pragma: no cover - 仅在缺少 torch 环境时兜底
    torch = None
    F = None


# DK 高斯平滑的固定 log_std，用于表达“确定性均值 + 小方差高斯”的设计。
DK_LOG_STD = -0.5
# CLQR DK 动作裁剪边界，直接沿用当前项目 Fused_CPRO.py 的默认口径。
CLQR_ACTION_MAX = 1.5
# MIMO DK 中 delay softmax 的温度/放大系数，越大越偏向高 delay 用户。
MIMO_DK_BETA = 8.5
# MIMO DK 中触发 urgency 增强的延迟阈值比例，按约束上限做缩放。
MIMO_DK_THRESHOLD_SCALE = 0.90
# MIMO DK 中的公平性混合权重，给所有用户保留一部分均匀分配。
MIMO_DK_FAIRNESS = 0.02
# MIMO DK 中对超阈值用户的额外权重提升。
MIMO_DK_THRESHOLD_BOOST = 0.30
# MIMO DK 中正则化因子的固定偏置下界。
MIMO_DK_REG_BIAS = 0.25
# MIMO 单用户功率的最大裁剪值。
MIMO_POWER_MAX = 2.5
# MIMO 正则化因子的最小值，避免数值退化。
MIMO_REG_MIN = 1e-6
# 动作变换的开区间下界；只用于 DK smoothing density 的反变换。
ACTION_EPS = 1e-6
# 反变换时的内部夹紧宽度，用于避免边界奇点。
ACTION_INVERSE_EPS = 1e-6
# MIMO 约束上限，用于 urgency 阈值判断。
MIMO_CONSTRAINT_LIMIT = 1.2
# 通用数值稳定项，避免除零或全零归一化。
EPS = 1e-8


def _as_numpy(array_like: Any) -> np.ndarray:
    return np.asarray(array_like, dtype=np.float64).reshape(-1)


def _normalize_scene_name(scene: str) -> str:
    scene_text = str(scene).strip().upper()
    if "MIMO" in scene_text:
        return "MIMO"
    if "CLQR" in scene_text:
        return "CLQR"
    raise ValueError("unsupported scene: {0}. expected MIMO or CLQR.".format(scene))


def _validate_action_dim(action: np.ndarray, expected_dim: int, name: str) -> np.ndarray:
    action_vec = _as_numpy(action)
    if action_vec.size != int(expected_dim):
        raise ValueError(
            "{0} action_dim mismatch: expected {1}, got {2}".format(
                name,
                int(expected_dim),
                int(action_vec.size),
            )
        )
    return action_vec


def _softplus_inverse(positive_torch: Any) -> Any:
    x = positive_torch.clamp_min(ACTION_INVERSE_EPS)
    return x + torch.log(-torch.expm1(-x))


def _mimo_inverse_action_and_log_det(action_torch: Any) -> tuple[Any, Any]:
    reg_mask = torch.zeros(action_torch.shape[-1], dtype=torch.bool, device=action_torch.device)
    reg_mask[-1] = True
    view_shape = [1] * action_torch.dim()
    view_shape[-1] = action_torch.shape[-1]
    reg_mask = reg_mask.view(*view_shape)

    power_action = action_torch.clamp(
        min=ACTION_EPS + ACTION_INVERSE_EPS,
        max=MIMO_POWER_MAX - ACTION_INVERSE_EPS,
    )
    power_z = (power_action - ACTION_EPS) / (MIMO_POWER_MAX - ACTION_EPS)
    raw_power = torch.logit(power_z)
    raw_reg = _softplus_inverse(action_torch - ACTION_EPS)
    raw_action = torch.where(reg_mask, raw_reg, raw_power)

    power_log_det = (
        torch.log(torch.tensor(MIMO_POWER_MAX - ACTION_EPS, dtype=action_torch.dtype, device=action_torch.device))
        + F.logsigmoid(raw_action)
        + F.logsigmoid(-raw_action)
    )
    reg_log_det = -F.softplus(-raw_action)
    log_det = torch.where(reg_mask, reg_log_det, power_log_det).sum(dim=-1)
    return raw_action, log_det


def _clqr_inverse_action_and_log_det(action_torch: Any) -> tuple[Any, Any]:
    scaled_action = (action_torch / CLQR_ACTION_MAX).clamp(
        min=-1.0 + ACTION_INVERSE_EPS,
        max=1.0 - ACTION_INVERSE_EPS,
    )
    raw_action = 0.5 * (torch.log1p(scaled_action) - torch.log1p(-scaled_action))
    log_det_per_dim = (
        torch.log(torch.tensor(CLQR_ACTION_MAX, dtype=action_torch.dtype, device=action_torch.device))
        + 2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
    )
    return raw_action, log_det_per_dim.sum(dim=-1)


class HeuristicGaussianPolicy:
    """轻量策略包装：确定性 DK 均值 + 固定高斯平滑。"""

    def __init__(
        self,
        mean_fn: Callable[[Any], np.ndarray],
        action_dim: int,
        device: str = "cpu",
        log_std: float = DK_LOG_STD,
        transform_kind: str = "mimo",
    ) -> None:
        self.mean_fn = mean_fn
        self.action_dim = int(action_dim)
        self.device = str(device)
        self.log_std = float(log_std)
        self.transform_kind = str(transform_kind).lower()

    def mean_action(self, state: Any) -> np.ndarray:
        action = self.mean_fn(state)
        return _validate_action_dim(action, self.action_dim, "heuristic policy")

    def sample_action(self, state: Any) -> np.ndarray:
        # 这里保留与现有 DK 策略一致的行为：采样阶段直接返回确定性均值动作。
        return self.mean_action(state)

    def log_prob_batch(self, states_torch: Any, actions_torch: Any) -> Any:
        """仅用于说明当前项目里 DK 平滑高斯的 log_prob 计算口径。"""

        if torch is None:
            raise ImportError("torch is required for log_prob_batch().")

        means = []
        for idx in range(states_torch.shape[0]):
            state_np = states_torch[idx].detach().cpu().numpy()
            means.append(self.mean_action(state_np))

        mu = torch.tensor(
            np.asarray(means, dtype=np.float64),
            dtype=torch.float,
            device=states_torch.device,
        )
        log_std = torch.full(
            (self.action_dim,),
            self.log_std,
            dtype=torch.float,
            device=states_torch.device,
        )
        if self.transform_kind == "clqr":
            raw_mu, _ = _clqr_inverse_action_and_log_det(mu)
            raw_action, log_det = _clqr_inverse_action_and_log_det(actions_torch)
        elif self.transform_kind == "mimo":
            raw_mu, _ = _mimo_inverse_action_and_log_det(mu)
            raw_action, log_det = _mimo_inverse_action_and_log_det(actions_torch)
        else:
            raise ValueError("unsupported DK transform_kind: {0}".format(self.transform_kind))

        std = torch.exp(log_std).view(1, -1).repeat(states_torch.shape[0], 1)
        dist = torch.distributions.normal.Normal(raw_mu, std)
        return dist.log_prob(raw_action).sum(dim=1) - log_det


def build_mimo_dk_action(
    state: Any,
    ue_num: int,
    beta: float = MIMO_DK_BETA,
    threshold_scale: float = MIMO_DK_THRESHOLD_SCALE,
    fairness: float = MIMO_DK_FAIRNESS,
    threshold_boost: float = MIMO_DK_THRESHOLD_BOOST,
    reg_bias: float = MIMO_DK_REG_BIAS,
    power_max: float = MIMO_POWER_MAX,
    reg_min: float = MIMO_REG_MIN,
    constraint_limit: float = MIMO_CONSTRAINT_LIMIT,
) -> np.ndarray:
    """按当前项目 MIMO DK 规则，从状态尾部 delay 向量生成动作。"""

    ue_num = int(ue_num)
    if ue_num <= 0:
        raise ValueError("ue_num must be positive.")

    state_vec = _as_numpy(state)
    if state_vec.size < ue_num:
        raise ValueError(
            "MIMO state length is too small for ue_num={0}: got state_dim={1}".format(
                ue_num,
                int(state_vec.size),
            )
        )

    delay_vec = np.maximum(state_vec[-ue_num:], 0.0)
    delay_norm = delay_vec / (np.sum(delay_vec) + EPS)

    logits = float(beta) * delay_norm
    logits = logits - np.max(logits)
    soft = np.exp(logits)
    soft = soft / (np.sum(soft) + EPS)

    urgency = (delay_vec > (float(threshold_scale) * float(constraint_limit))).astype(np.float64)
    urgency = urgency / max(float(ue_num), 1.0)

    share = (
        (1.0 - float(fairness)) * soft
        + float(fairness) * (1.0 / float(ue_num))
        + float(threshold_boost) * urgency
    )
    share = np.clip(share, EPS, None)
    share = share / np.sum(share)

    power = np.clip(float(power_max) * share * float(ue_num), EPS, float(power_max))
    reg = float(max(float(reg_bias), float(reg_min)))
    return np.concatenate((power.astype(np.float64), np.asarray([reg], dtype=np.float64)), axis=0)


def build_clqr_stabilizing_gain(
    A: Any,
    B: Any,
    gain_scale: float = 0.25,
    seed: int = 0,
) -> np.ndarray:
    """复刻当前 CLQR DK 的启发式稳定化增益构造方式。"""

    a_mat = np.asarray(A, dtype=np.float64)
    b_mat = np.asarray(B, dtype=np.float64)

    if a_mat.ndim != 2 or a_mat.shape[0] != a_mat.shape[1]:
        raise ValueError("A must be a square 2D array.")
    if b_mat.ndim != 2 or b_mat.shape[0] != a_mat.shape[0]:
        raise ValueError("B must be a 2D array with the same row dimension as A.")

    state_dim = int(a_mat.shape[0])
    action_dim = int(b_mat.shape[1])
    rng = np.random.default_rng(int(seed))
    k = rng.normal(0.0, 0.2, size=(action_dim, state_dim))

    a_cl = a_mat - b_mat @ k
    spectral = float(np.max(np.abs(np.linalg.eigvals(a_cl))))
    if spectral < 1e-6:
        spectral = 1.0

    scale = min(1.0, 0.95 / spectral)
    return float(gain_scale) * scale * k


def build_clqr_dk_action(
    state: Any,
    gain: Any,
    action_max: float = CLQR_ACTION_MAX,
) -> np.ndarray:
    """按当前项目 CLQR DK 规则，使用 -(Kx) 并做逐维裁剪。"""

    state_vec = _as_numpy(state)
    gain_mat = np.asarray(gain, dtype=np.float64)

    if gain_mat.ndim != 2:
        raise ValueError("gain must be a 2D array.")
    if gain_mat.shape[1] != state_vec.size:
        raise ValueError(
            "CLQR gain/state mismatch: gain expects state_dim={0}, got {1}".format(
                int(gain_mat.shape[1]),
                int(state_vec.size),
            )
        )

    action = -(gain_mat @ state_vec)
    return np.clip(action, -float(action_max), float(action_max)).astype(np.float64)


def build_mimo_dk_policy(
    action_dim: int,
    ue_num: int,
    device: str = "cpu",
    log_std: float = DK_LOG_STD,
    beta: float = MIMO_DK_BETA,
    threshold_scale: float = MIMO_DK_THRESHOLD_SCALE,
    fairness: float = MIMO_DK_FAIRNESS,
    threshold_boost: float = MIMO_DK_THRESHOLD_BOOST,
    reg_bias: float = MIMO_DK_REG_BIAS,
    power_max: float = MIMO_POWER_MAX,
    reg_min: float = MIMO_REG_MIN,
    constraint_limit: float = MIMO_CONSTRAINT_LIMIT,
) -> HeuristicGaussianPolicy:
    expected_action_dim = int(ue_num) + 1
    if int(action_dim) != expected_action_dim:
        raise ValueError(
            "MIMO action_dim mismatch: expected ue_num + 1 = {0}, got {1}".format(
                expected_action_dim,
                int(action_dim),
            )
        )

    def mimo_mean(state: Any) -> np.ndarray:
        return build_mimo_dk_action(
            state=state,
            ue_num=ue_num,
            beta=beta,
            threshold_scale=threshold_scale,
            fairness=fairness,
            threshold_boost=threshold_boost,
            reg_bias=reg_bias,
            power_max=power_max,
            reg_min=reg_min,
            constraint_limit=constraint_limit,
        )

    return HeuristicGaussianPolicy(
        mean_fn=mimo_mean,
        action_dim=action_dim,
        device=device,
        log_std=log_std,
        transform_kind="mimo",
    )


def build_clqr_dk_policy(
    action_dim: int,
    A: Any,
    B: Any,
    device: str = "cpu",
    log_std: float = DK_LOG_STD,
    gain_scale: float = 0.25,
    seed: int = 0,
    action_max: float = CLQR_ACTION_MAX,
) -> HeuristicGaussianPolicy:
    b_mat = np.asarray(B, dtype=np.float64)
    if b_mat.ndim != 2:
        raise ValueError("B must be a 2D array.")
    expected_action_dim = int(b_mat.shape[1])
    if int(action_dim) != expected_action_dim:
        raise ValueError(
            "CLQR action_dim mismatch: expected B.shape[1] = {0}, got {1}".format(
                expected_action_dim,
                int(action_dim),
            )
        )

    gain = build_clqr_stabilizing_gain(A=A, B=B, gain_scale=gain_scale, seed=seed)

    def clqr_mean(state: Any) -> np.ndarray:
        return build_clqr_dk_action(state=state, gain=gain, action_max=action_max)

    policy = HeuristicGaussianPolicy(
        mean_fn=clqr_mean,
        action_dim=action_dim,
        device=device,
        log_std=log_std,
        transform_kind="clqr",
    )
    policy.gain = gain
    return policy


def build_dk_policy(scene: str, **kwargs: Any) -> HeuristicGaussianPolicy:
    """统一入口：根据场景名构造 MIMO 或 CLQR DK 策略。"""

    scene_name = _normalize_scene_name(scene)
    if scene_name == "MIMO":
        return build_mimo_dk_policy(**kwargs)
    if scene_name == "CLQR":
        return build_clqr_dk_policy(**kwargs)
    raise ValueError("unsupported scene: {0}".format(scene))


__all__ = [
    "ACTION_EPS",
    "ACTION_INVERSE_EPS",
    "CLQR_ACTION_MAX",
    "DK_LOG_STD",
    "EPS",
    "HeuristicGaussianPolicy",
    "MIMO_CONSTRAINT_LIMIT",
    "MIMO_DK_BETA",
    "MIMO_DK_FAIRNESS",
    "MIMO_DK_REG_BIAS",
    "MIMO_DK_THRESHOLD_BOOST",
    "MIMO_DK_THRESHOLD_SCALE",
    "MIMO_POWER_MAX",
    "MIMO_REG_MIN",
    "build_clqr_dk_action",
    "build_clqr_dk_policy",
    "build_clqr_stabilizing_gain",
    "build_dk_policy",
    "build_mimo_dk_action",
    "build_mimo_dk_policy",
]
