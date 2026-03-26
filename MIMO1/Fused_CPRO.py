import copy
import os

import numpy as np
import torch

try:
    import cvxpy as cp
except Exception:
    cp = None

from buffer import DataStorage
from critic_opt import Critic
from environment import Environment_CLQR, Environment_MIMO
from model import GaussianPolicy_CLQR, GaussianPolicy_MIMO


# 策略库配置：每个场景固定使用 1 个 DK 策略 + 2 个镜像 SLDAC 策略。
NUM_SLDAC_LIBRARY_POLICIES = 2
# rho 的单纯形下界，避免对数与数值更新退化。
RHO_MIN_NEW_ACTOR = 0.2
RHO_MIN_OLD_POLICY = 1e-4
# actor/rho 梯度阶段中 offline 样本占比。
# xi 的默认值：表示额外离线样本量相对在线样本量的系数。
# xi 的默认值：表示 offline 分支在 actor/rho 梯度中的权重。
DEFAULT_OFFLINE_WEIGHT = 0.5
# 每个 old policy 的离线轨迹长度，默认与参考窗口 T 对齐。
OFFLINE_STEPS_MULTIPLIER = 1
# DK 策略用于 log_prob 评估时的固定高斯标准差。
DK_LOG_STD = -0.5
# CLQR DK 策略的动作裁剪边界，沿用当前项目 clqr_env 的口径。
CLQR_ACTION_MAX = 1.5
# MIMO DK old_heavy 口径参数。
MIMO_DK_BETA = 8.5
MIMO_DK_THRESHOLD_SCALE = 0.90
MIMO_DK_FAIRNESS = 0.02
MIMO_DK_THRESHOLD_BOOST = 0.30
MIMO_DK_REG_BIAS = 0.25
MIMO_POWER_MAX = 2.5
MIMO_REG_MIN = 1e-6
MIMO_CONSTRAINT_LIMIT = 1.2
EPS = 1e-8
CHECKPOINT_FILE_TEMPLATE = "episode_{0:04d}.pt"
CHECKPOINT_FINAL_FILE_TEMPLATE = "episode_{0:04d}_final.pt"
RHO_SCHEDULER_POWER = "power"
RHO_SCHEDULER_COSINE_RESTART_DECAY = "cosine_restart_decay"


def _format_seed_dir(seed):
    return "seed_{0}".format(int(seed))


def _as_numpy(array_like):
    return np.asarray(array_like, dtype=np.float64).reshape(-1)


def _set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _is_mimo(example_name):
    return "MIMO" in str(example_name)


def _build_scene(example_name, seed, device, policy_batch_size):
    if _is_mimo(example_name):
        nt, ue_num = 8, 4
        state_dim = 2 * ue_num * nt + ue_num
        action_dim = ue_num + 1
        env = Environment_MIMO(seed=int(seed), Nt=nt, UE_num=ue_num)
        constraint_dim = ue_num
        constr_lim = np.full((constraint_dim,), MIMO_CONSTRAINT_LIMIT, dtype=np.float64)
        actor = GaussianPolicy_MIMO(state_dim, action_dim, device, int(policy_batch_size))
    else:
        state_dim = 15
        action_dim = 4
        env = Environment_CLQR(seed=int(seed), state_dim=state_dim, action_dim=action_dim)
        constraint_dim = 1
        constr_lim = 380.0 * np.ones((constraint_dim,), dtype=np.float64)
        actor = GaussianPolicy_CLQR(state_dim, action_dim, device, int(policy_batch_size))
    return env, actor, state_dim, action_dim, constraint_dim, constr_lim


def _make_buffer(example_name, t_horizon, window, num_new_data, state_dim, action_dim, constraint_dim):
    if _is_mimo(example_name):
        return DataStorage(t_horizon, num_new_data, state_dim, action_dim, constraint_dim, window, 1)
    return DataStorage(t_horizon, window, num_new_data, 1, state_dim, action_dim, constraint_dim)


def _build_costs(reward, info, constraint_dim, constr_lim):
    costs = np.zeros((constraint_dim + 1,), dtype=np.float64)
    costs[0] = float(reward)
    for idx in range(1, constraint_dim + 1):
        costs[idx] = float(info.get(f"cost_{idx}", info.get("cost", 0.0)) - constr_lim[idx - 1])
    return costs


def _mean_action(actor, state):
    actor.net.eval()
    state_torch = torch.tensor(state, dtype=torch.float, device=actor.device)
    with torch.no_grad():
        mu = actor.net(state_torch)
    return mu.detach().cpu().numpy().reshape(-1)


def _sample_action(actor, state):
    action = actor.sample_action(state)
    return np.asarray(action, dtype=np.float64).reshape(-1)


def _log_prob_batch(actor, states_torch, actions_torch, require_grad):
    actor.net.train()
    with torch.set_grad_enabled(require_grad):
        mu = actor.net(states_torch)
        log_std = actor.log_std
        if require_grad:
            log_std = log_std.detach().clone().requires_grad_(True)
        else:
            log_std = log_std.detach()
        std = torch.exp(log_std).view(1, -1).repeat(states_torch.shape[0], 1)
        dist = torch.distributions.normal.Normal(mu, std)
        log_prob = dist.log_prob(actions_torch).sum(dim=1)
    return log_prob, log_std


def _zero_actor_extra_grad(actor):
    actor.zero_grad()
    if getattr(actor, "log_std", None) is not None and actor.log_std.grad is not None:
        actor.log_std.grad.zero_()


def _flatten_actor(actor):
    params = []
    for para in actor.net.parameters():
        params.append(para.data.view(-1))
    params.append(actor.log_std.detach().view(-1))
    return torch.cat(params).cpu().numpy().astype(np.float64, copy=False)


def _flatten_actor_grad(actor, log_std_grad):
    grads = []
    for para in actor.net.parameters():
        if para.grad is None:
            grads.append(torch.zeros_like(para).view(-1))
        else:
            grads.append(para.grad.view(-1))
    if log_std_grad is None:
        grads.append(torch.zeros_like(actor.log_std).view(-1))
    else:
        grads.append(log_std_grad.view(-1))
    return torch.cat(grads).detach().cpu().numpy().astype(np.float64, copy=False)


def _merge_log_std_grads(*grads):
    merged = None
    for grad in grads:
        if grad is None:
            continue
        if merged is None:
            merged = grad.detach().clone()
        else:
            merged = merged + grad.detach()
    return merged


def _set_actor_from_flat(actor, flat_params):
    flat = np.asarray(flat_params, dtype=np.float64).reshape(-1)
    offset = 0
    with torch.no_grad():
        for para in actor.net.parameters():
            numel = para.numel()
            block = torch.tensor(flat[offset:offset + numel], dtype=para.dtype, device=para.device)
            para.copy_(block.view_as(para))
            offset += numel
    log_std_np = flat[offset:offset + actor.action_dim]
    actor.log_std = torch.tensor(log_std_np, dtype=torch.float, device=actor.device)


def _rms_drift(new_vec, old_vec):
    diff = np.asarray(new_vec, dtype=np.float64).reshape(-1) - np.asarray(old_vec, dtype=np.float64).reshape(-1)
    if diff.size <= 0:
        return 0.0
    return float(np.linalg.norm(diff) / np.sqrt(diff.size))


def _clone_actor(actor):
    cloned = copy.deepcopy(actor)
    cloned.log_std = cloned.log_std.detach().clone()
    for para in cloned.net.parameters():
        para.requires_grad_(False)
    return cloned


class FrozenActorPolicy:
    def __init__(self, actor):
        self.actor = _clone_actor(actor)

    def sample_action(self, state):
        return _sample_action(self.actor, state)

    def log_prob_batch(self, states_torch, actions_torch):
        with torch.no_grad():
            mu = self.actor.net(states_torch)
            std = torch.exp(self.actor.log_std.detach()).view(1, -1).repeat(states_torch.shape[0], 1)
            dist = torch.distributions.normal.Normal(mu, std)
            return dist.log_prob(actions_torch).sum(dim=1)


class HeuristicGaussianPolicy:
    def __init__(self, mean_fn, action_dim, device, log_std=DK_LOG_STD):
        self.mean_fn = mean_fn
        self.action_dim = int(action_dim)
        self.device = device
        self.log_std = torch.full((self.action_dim,), float(log_std), dtype=torch.float, device=device)

    def mean_action(self, state):
        return np.asarray(self.mean_fn(state), dtype=np.float64).reshape(-1)

    def sample_action(self, state):
        return self.mean_action(state)

    def log_prob_batch(self, states_torch, actions_torch):
        means = []
        for idx in range(states_torch.shape[0]):
            means.append(self.mean_action(states_torch[idx].detach().cpu().numpy()))
        mu = torch.tensor(np.asarray(means, dtype=np.float64), dtype=torch.float, device=states_torch.device)
        std = torch.exp(self.log_std).view(1, -1).repeat(states_torch.shape[0], 1)
        dist = torch.distributions.normal.Normal(mu, std)
        return dist.log_prob(actions_torch).sum(dim=1)


def _clqr_stabilizing_gain(env, gain_scale=0.25, seed=0):
    rng = np.random.default_rng(int(seed))
    k = rng.normal(0.0, 0.2, size=(env.action_dim, env.state_dim))
    a_cl = env.A - env.B @ k
    spectral = float(np.max(np.abs(np.linalg.eigvals(a_cl))))
    if spectral < 1e-6:
        spectral = 1.0
    scale = min(1.0, 0.95 / spectral)
    return gain_scale * scale * k


def _build_dk_policy(example_name, env, device, seed):
    if _is_mimo(example_name):
        ue_num = env.UE_num

        def mimo_mean(state):
            delay_vec = np.maximum(_as_numpy(state)[-ue_num:], 0.0)
            delay_norm = delay_vec / (np.sum(delay_vec) + EPS)
            logits = MIMO_DK_BETA * delay_norm
            logits = logits - np.max(logits)
            soft = np.exp(logits)
            soft = soft / (np.sum(soft) + EPS)
            urgency = (delay_vec > (MIMO_DK_THRESHOLD_SCALE * MIMO_CONSTRAINT_LIMIT)).astype(np.float64)
            urgency = urgency / max(float(ue_num), 1.0)
            share = (
                (1.0 - MIMO_DK_FAIRNESS) * soft
                + MIMO_DK_FAIRNESS * (1.0 / float(ue_num))
                + MIMO_DK_THRESHOLD_BOOST * urgency
            )
            share = np.clip(share, EPS, None)
            share = share / np.sum(share)
            power = np.clip(MIMO_POWER_MAX * share * float(ue_num), EPS, MIMO_POWER_MAX)
            reg = float(max(MIMO_DK_REG_BIAS, MIMO_REG_MIN))
            return np.concatenate((power.astype(np.float64), np.asarray([reg], dtype=np.float64)), axis=0)

        return HeuristicGaussianPolicy(mimo_mean, env.action_dim, device)

    gain = _clqr_stabilizing_gain(env, gain_scale=0.25, seed=int(seed) + 17)

    def clqr_mean(state):
        action = -(gain @ _as_numpy(state))
        return np.clip(action, -CLQR_ACTION_MAX, CLQR_ACTION_MAX)

    return HeuristicGaussianPolicy(clqr_mean, env.action_dim, device)


def _policy_rollout_dataset(example_name, policy, steps, seed, device):
    env, _, state_dim, action_dim, constraint_dim, constr_lim = _build_scene(example_name, seed, device, max(1, int(steps)))
    observation = env.reset()
    states = np.zeros((steps, state_dim), dtype=np.float64)
    actions = np.zeros((steps, action_dim), dtype=np.float64)
    for idx in range(int(steps)):
        state = observation
        action = policy.sample_action(state)
        next_state, reward, done, info = env.step(action)
        _ = _build_costs(reward, info, constraint_dim, constr_lim)
        states[idx] = state
        actions[idx] = np.asarray(action, dtype=np.float64).reshape(-1)
        observation = next_state
        if done:
            observation = env.reset()
    return {"state": states, "action": actions}


def _sample_offline_batch(datasets, batch_size, state_dim, action_dim):
    batch_size = int(batch_size)
    if batch_size <= 0 or len(datasets) <= 0:
        return (
            np.zeros((0, state_dim), dtype=np.float64),
            np.zeros((0, action_dim), dtype=np.float64),
        )
    states = np.zeros((batch_size, state_dim), dtype=np.float64)
    actions = np.zeros((batch_size, action_dim), dtype=np.float64)
    for idx in range(batch_size):
        ds = datasets[np.random.randint(len(datasets))]
        row = np.random.randint(ds["state"].shape[0])
        states[idx] = ds["state"][row]
        actions[idx] = ds["action"][row]
    return states, actions


def _coerce_rho_lower_bounds(rho_lower_bounds, simplex_dim):
    simplex_dim = int(simplex_dim)
    if simplex_dim <= 0:
        raise ValueError("simplex_dim must be positive")
    arr = np.asarray(rho_lower_bounds, dtype=np.float64).reshape(-1)
    if arr.size == 1:
        arr = np.full((simplex_dim,), float(arr[0]), dtype=np.float64)
    elif arr.size != simplex_dim:
        raise ValueError(
            "rho_lower_bounds size mismatch. expected {0}, got {1}".format(simplex_dim, arr.size)
        )
    if (not np.isfinite(arr).all()) or np.any(arr < 0.0):
        raise ValueError("rho_lower_bounds must be finite and non-negative")
    if float(np.sum(arr)) > 1.0 + EPS:
        raise ValueError("rho lower bounds are infeasible: sum={0:.6f} > 1".format(float(np.sum(arr))))
    return arr


def _build_rho_lower_bounds(simplex_dim, rho_min_new_actor=RHO_MIN_NEW_ACTOR, rho_min_old_policy=RHO_MIN_OLD_POLICY):
    simplex_dim = int(simplex_dim)
    if simplex_dim <= 0:
        raise ValueError("simplex_dim must be positive")
    lower_bounds = np.full((simplex_dim,), float(rho_min_old_policy), dtype=np.float64)
    lower_bounds[0] = float(rho_min_new_actor)
    return _coerce_rho_lower_bounds(lower_bounds, simplex_dim)


def _resolve_rho_lower_bounds(args, simplex_dim):
    return _build_rho_lower_bounds(
        simplex_dim,
        rho_min_new_actor=float(getattr(args, "rho_min_new_actor", RHO_MIN_NEW_ACTOR)),
        rho_min_old_policy=float(getattr(args, "rho_min_old_policy", RHO_MIN_OLD_POLICY)),
    )


def _project_simplex(values, simplex_sum):
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError("values must be non-empty")
    target_sum = float(simplex_sum)
    if target_sum < -EPS:
        raise ValueError("simplex_sum must be non-negative")
    if target_sum <= EPS:
        return np.zeros_like(arr)
    u = np.sort(arr)[::-1]
    cssv = np.cumsum(u) - target_sum
    ind = np.arange(1, arr.size + 1, dtype=np.float64)
    cond = u - cssv / ind > 0
    if np.any(cond):
        rho_idx = np.nonzero(cond)[0][-1]
        theta = cssv[rho_idx] / float(rho_idx + 1)
    else:
        theta = cssv[-1] / float(arr.size)
    return np.maximum(arr - theta, 0.0)


def _normalize_simplex(rho, rho_lower_bounds):
    arr = np.asarray(rho, dtype=np.float64).reshape(-1)
    if arr.size <= 0:
        raise ValueError("rho must be non-empty")
    lower_bounds = _coerce_rho_lower_bounds(rho_lower_bounds, arr.size)
    residual = 1.0 - float(np.sum(lower_bounds))
    projected = _project_simplex(arr - lower_bounds, residual)
    return lower_bounds + projected


def _build_simplex_constraints(constr, paras_cvx, simplex_dim, rho_lower_bounds):
    if simplex_dim is None or int(simplex_dim) <= 0:
        return constr
    lower_bounds = _coerce_rho_lower_bounds(rho_lower_bounds, int(simplex_dim))
    constr += [paras_cvx[:simplex_dim] >= lower_bounds, cp.sum(paras_cvx[:simplex_dim]) == 1]
    return constr


def _get_xi(update_index, decay_pow, xi0):
    xi = float(xi0) / ((int(update_index) + 1) ** float(decay_pow))
    return min(max(xi, 0.0), 1.0)


def _build_policy_gradient_batch_impl(
    online_state_batch,
    online_action_batch,
    offline_datasets,
    xi,
    grad_t,
    state_dim,
    action_dim,
    use_offline_data,
):
    _ = xi
    online_states = np.asarray(online_state_batch, dtype=np.float64).copy()
    online_actions = np.asarray(online_action_batch, dtype=np.float64).copy()
    grad_t = int(grad_t)
    if grad_t <= 0:
        raise ValueError("grad_t must be a positive integer")
    if online_states.shape[0] < grad_t or online_actions.shape[0] < grad_t:
        raise ValueError("online policy-gradient batch is shorter than grad_t")
    online_states = online_states[-grad_t:]
    online_actions = online_actions[-grad_t:]
    if not use_offline_data:
        return online_states, online_actions

    n_online = grad_t
    n_offline = grad_t
    total_batch = n_online + n_offline
    fused_state_batch = np.zeros((total_batch, state_dim), dtype=np.float64)
    fused_action_batch = np.zeros((total_batch, action_dim), dtype=np.float64)
    fused_state_batch[:n_online] = online_states
    fused_action_batch[:n_online] = online_actions
    if n_offline > 0:
        offline_states, offline_actions = _sample_offline_batch(offline_datasets, n_offline, state_dim, action_dim)
        fused_state_batch[n_online:] = offline_states
        fused_action_batch[n_online:] = offline_actions
    return fused_state_batch, fused_action_batch


def _get_rho_scheduler_mode(args):
    mode = str(getattr(args, "rho_scheduler", RHO_SCHEDULER_POWER)).strip().lower()
    if not mode:
        return RHO_SCHEDULER_POWER
    if mode not in (RHO_SCHEDULER_POWER, RHO_SCHEDULER_COSINE_RESTART_DECAY):
        raise ValueError("unsupported rho_scheduler: {0}".format(mode))
    return mode


def _build_rho_scheduler_config(args, beta_actor_pow):
    mode = _get_rho_scheduler_mode(args)
    if mode == RHO_SCHEDULER_POWER:
        beta_rho_pow = float(getattr(args, "beta_rho_pow", beta_actor_pow))
        if beta_rho_pow <= beta_actor_pow:
            raise ValueError(
                "beta_rho_pow must be greater than beta_actor_pow. got beta_actor_pow={0}, beta_rho_pow={1}".format(
                    beta_actor_pow,
                    beta_rho_pow,
                )
            )
        return {
            "mode": mode,
            "beta_rho_pow": beta_rho_pow,
            "xi_decay_pow": beta_rho_pow,
        }

    beta_peak_init = float(getattr(args, "rho_beta_peak_init"))
    beta_peak_final_ratio = float(getattr(args, "rho_beta_peak_final_ratio"))
    beta_min = float(getattr(args, "rho_beta_min"))
    restart_rounds = int(getattr(args, "rho_restart_rounds"))
    period_mult = int(getattr(args, "rho_period_mult"))
    total_updates = int(getattr(args, "num_update_time"))
    xi_decay_pow = float(getattr(args, "xi_decay_pow", beta_actor_pow))

    if beta_min <= 0.0:
        raise ValueError("rho_beta_min must be positive. got rho_beta_min={0}".format(beta_min))
    if beta_peak_init < beta_min:
        raise ValueError(
            "rho_beta_peak_init must be greater than or equal to rho_beta_min. got rho_beta_peak_init={0}, rho_beta_min={1}".format(
                beta_peak_init,
                beta_min,
            )
        )
    if beta_peak_init > 1.0:
        raise ValueError("rho_beta_peak_init must be less than or equal to 1.0. got {0}".format(beta_peak_init))
    if (beta_peak_final_ratio <= 0.0) or (beta_peak_final_ratio > 1.0):
        raise ValueError(
            "rho_beta_peak_final_ratio must be in (0, 1]. got {0}".format(beta_peak_final_ratio)
        )
    if restart_rounds <= 0:
        raise ValueError("rho_restart_rounds must be a positive integer. got {0}".format(restart_rounds))
    if period_mult < 1:
        raise ValueError("rho_period_mult must be an integer greater than or equal to 1. got {0}".format(period_mult))
    if total_updates <= 0:
        raise ValueError("num_update_time must be a positive integer. got {0}".format(total_updates))

    cycle_count = min(restart_rounds, total_updates)
    periods = np.ones((cycle_count,), dtype=np.int32)
    remaining_updates = total_updates - cycle_count
    if remaining_updates > 0:
        weights = np.power(float(period_mult), np.arange(cycle_count, dtype=np.float64))
        weight_sum = float(np.sum(weights))
        extra_periods = np.floor((remaining_updates * weights) / weight_sum).astype(np.int32)
        periods += extra_periods
        remainder = int(total_updates - int(np.sum(periods)))
        for idx in range(remainder):
            periods[-1 - (idx % cycle_count)] += 1

    return {
        "mode": mode,
        "beta_peak_init": beta_peak_init,
        "beta_peak_final_ratio": beta_peak_final_ratio,
        "beta_min": beta_min,
        "periods": periods,
        "period_mult": period_mult,
        "xi_decay_pow": xi_decay_pow,
    }


def _get_rho_beta(update_index, scheduler_config):
    if scheduler_config["mode"] == RHO_SCHEDULER_POWER:
        return 1.0 / ((int(update_index) + 1) ** float(scheduler_config["beta_rho_pow"]))

    periods = np.asarray(scheduler_config["periods"], dtype=np.int32).reshape(-1)
    restart_index = 0
    local_index = int(update_index)
    period = int(periods[-1])
    for idx, current_period in enumerate(periods):
        period = int(current_period)
        if local_index < period:
            restart_index = idx
            break
        restart_index += 1
        local_index -= period
    if restart_index >= periods.size:
        restart_index = periods.size - 1
        period = int(periods[-1])
        local_index = max(0, period - 1)

    beta_min = float(scheduler_config["beta_min"])
    if periods.size <= 1:
        peak_ratio = 1.0
    else:
        peak_ratio = float(scheduler_config["beta_peak_final_ratio"]) ** (float(restart_index) / float(periods.size - 1))
    beta_peak = float(scheduler_config["beta_peak_init"]) * peak_ratio
    beta_peak = max(beta_peak, beta_min)
    if period <= 1:
        return beta_peak

    phase = float(local_index) / float(period - 1)
    cosine_weight = 0.5 * (1.0 + np.cos(np.pi * phase))
    return beta_min + (beta_peak - beta_min) * cosine_weight


def _solve_problem(prob):
    if cp is None:
        raise ModuleNotFoundError("cvxpy is required for Fused_CPRO.")
    candidates = [cp.MOSEK, cp.OSQP, cp.ECOS, cp.SCS]
    last_err = None
    for solver in candidates:
        try:
            prob.solve(solver=solver, warm_start=True)
        except Exception as ex:
            last_err = ex
            continue
        if prob.status in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE):
            return prob.status
    if last_err is not None:
        print(f"cvxpy fallback failed: {last_err}")
    prob.solve(warm_start=True)
    return prob.status


def _policy_update(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost, simplex_dim=None, rho_lower_bounds=0.0):
    if cp is None:
        raise ModuleNotFoundError("cvxpy is required for Fused_CPRO.")
    if (not np.isfinite(func_value_np).all()) or (not np.isfinite(grad_np).all()) or (not np.isfinite(paras_t_np).all()):
        print("policy update skipped: non-finite func/grad/params")
        return paras_t_np
    x_val, paras_bar, _ = _feasible_update(
        func_value_np,
        grad_np,
        paras_t_np,
        tau_cost,
        simplex_dim=simplex_dim,
        rho_lower_bounds=rho_lower_bounds,
    )
    if x_val == np.inf or paras_bar is None or (not np.isfinite(paras_bar).all()):
        return paras_t_np
    if x_val <= 0:
        paras_obj, _ = _objective_update(
            func_value_np,
            grad_np,
            paras_t_np,
            tau_reward=tau_reward,
            tau_cost=tau_cost,
            simplex_dim=simplex_dim,
            rho_lower_bounds=rho_lower_bounds,
        )
        if paras_obj is not None and np.isfinite(paras_obj).all():
            return paras_obj
    return paras_bar


def _objective_update(func_value_np, grad_np, paras_t_np, tau_reward, tau_cost, simplex_dim=None, rho_lower_bounds=0.0):
    m = grad_np.shape[0] - 1
    n = grad_np.shape[1]
    tau_np = tau_cost * np.ones((m + 1,), dtype=np.float64)
    tau_np[0] = tau_reward
    paras_cvx = cp.Variable(shape=(n,))
    obj = func_value_np[0] + grad_np[0].T @ (paras_cvx - paras_t_np) + tau_np[0] * cp.sum_squares(paras_cvx - paras_t_np)
    constr = []
    constr = _build_simplex_constraints(constr, paras_cvx, simplex_dim, rho_lower_bounds)
    for idx in range(1, m + 1):
        constr += [func_value_np[idx] + grad_np[idx].T @ (paras_cvx - paras_t_np) + tau_np[idx] * cp.sum_squares(paras_cvx - paras_t_np) <= 0]
    prob = cp.Problem(cp.Minimize(obj), constr)
    status = _solve_problem(prob)
    return paras_cvx.value, status


def _feasible_update(func_value_np, grad_np, paras_t_np, tau_cost, simplex_dim=None, rho_lower_bounds=0.0):
    m = grad_np.shape[0] - 1
    n = grad_np.shape[1]
    func_value = func_value_np[1:]
    grad_val = grad_np[1:]
    tau_np = tau_cost * np.ones((m,), dtype=np.float64)
    paras_cvx = cp.Variable(shape=(n,))
    x_cvx = cp.Variable()
    obj = x_cvx
    constr = []
    constr = _build_simplex_constraints(constr, paras_cvx, simplex_dim, rho_lower_bounds)
    for idx in range(m):
        constr += [func_value[idx] + grad_val[idx].T @ (paras_cvx - paras_t_np) + tau_np[idx] * cp.sum_squares(paras_cvx - paras_t_np) <= x_cvx]
    prob = cp.Problem(cp.Minimize(obj), constr)
    status = _solve_problem(prob)
    return prob.value, paras_cvx.value, status


def _build_mixture_log_prob(state_batch_torch, action_batch_torch, actor_new, old_policies, rho, rho_lower_bounds, rho_torch=None):
    if rho_torch is None:
        rho_torch = torch.tensor(rho, dtype=torch.float, device=state_batch_torch.device, requires_grad=True)
    log_prob_new, log_std_leaf = _log_prob_batch(actor_new, state_batch_torch, action_batch_torch, require_grad=True)
    log_prob_list = [log_prob_new]
    for policy in old_policies:
        log_prob_list.append(policy.log_prob_batch(state_batch_torch, action_batch_torch))
    log_pi = torch.stack(log_prob_list, dim=1)
    lower_bounds = _coerce_rho_lower_bounds(rho_lower_bounds, len(rho))
    rho_safe = torch.maximum(
        rho_torch,
        torch.tensor(lower_bounds, dtype=rho_torch.dtype, device=state_batch_torch.device),
    )
    log_mix = torch.logsumexp(log_pi + torch.log(rho_safe).view(1, -1), dim=1)
    return log_mix, rho_torch, log_std_leaf


def _q_head_normalize(q_hat):
    q_hat = np.asarray(q_hat, dtype=np.float64)
    if q_hat.ndim != 2 or q_hat.shape[0] <= 0:
        return q_hat
    out = q_hat.copy()
    reward_std = np.std(out[:, 0]) + 1e-6
    out[:, 0] = (out[:, 0] - np.mean(out[:, 0])) / reward_std
    for idx in range(1, out.shape[1]):
        out[:, idx] = (out[:, idx] - np.mean(out[:, idx])) / reward_std
    return out


def _blend_online_offline_loss(loss_online, loss_offline, xi, use_offline_data):
    if (not use_offline_data) or loss_offline is None:
        return loss_online
    xi = float(xi)
    return (1.0 - xi) * loss_online + xi * loss_offline


def _prcrl_q_head_normalize(q_hat):
    q_hat = np.asarray(q_hat, dtype=np.float64)
    if q_hat.ndim != 2 or q_hat.shape[0] <= 0:
        return q_hat
    out = q_hat.copy()
    reward_std = np.std(out[:, 0]) + 1e-6
    out[:, 0] = (out[:, 0] - np.mean(out[:, 0])) / reward_std
    for idx in range(1, out.shape[1]):
        out[:, idx] = out[:, idx] - np.mean(out[:, idx])
    return out


def _prcrl_window_q_hat(costs_batch, func_value, t_horizon):
    costs_arr = np.asarray(costs_batch, dtype=np.float64)
    if costs_arr.shape[0] < (2 * int(t_horizon)):
        raise ValueError(
            "PRCRL requires at least 2T samples in the buffer. got {0}, need {1}".format(
                int(costs_arr.shape[0]),
                int(2 * t_horizon),
            )
        )
    q_hat = np.zeros((int(t_horizon), costs_arr.shape[1]), dtype=np.float64)
    for idx in range(int(t_horizon)):
        costs_tmp = costs_arr[idx + 1 : idx + 1 + int(t_horizon)]
        q_hat[idx] = np.sum(costs_tmp, axis=0) - float(t_horizon) * np.asarray(func_value, dtype=np.float64)
    return q_hat


def _normalize_run_tags(run_tags):
    if run_tags is None:
        return []
    if isinstance(run_tags, str):
        return [tag.strip() for tag in run_tags.split(",") if tag.strip()]
    if isinstance(run_tags, (list, tuple)):
        normalized = []
        for tag in run_tags:
            text = str(tag).strip()
            if text:
                normalized.append(text)
        return normalized
    raise TypeError("old_policy_run_tags must be a comma-separated string or a sequence of strings.")


def _get_checkpoint_root(args):
    root = getattr(args, "checkpoint_root", os.path.join("checkpoints", "SLDAC"))
    if not root:
        root = os.path.join("checkpoints", "SLDAC")
    root = str(root)
    if not os.path.isabs(root):
        root = os.path.join(os.getcwd(), root)
    return root


def _resolve_sldac_checkpoint_path(args, example_name, run_tag, pretrain_episode, seed):
    checkpoint_dir = os.path.join(
        _get_checkpoint_root(args),
        str(example_name),
        str(run_tag),
        _format_seed_dir(seed),
    )
    candidates = [
        os.path.join(checkpoint_dir, CHECKPOINT_FILE_TEMPLATE.format(int(pretrain_episode))),
        os.path.join(checkpoint_dir, CHECKPOINT_FINAL_FILE_TEMPLATE.format(int(pretrain_episode))),
    ]
    for checkpoint_path in candidates:
        if os.path.exists(checkpoint_path):
            return checkpoint_path
    raise FileNotFoundError(
        "SLDAC checkpoint not found for run_tag={0}, seed={1}, pretrain_episode={2}. tried: {3}".format(
            run_tag,
            int(seed),
            int(pretrain_episode),
            ", ".join(candidates),
        )
    )


def _load_old_policy_from_checkpoint(
    args,
    example_name,
    run_tag,
    pretrain_episode,
    seed,
    device,
    policy_batch_size,
    state_dim,
    action_dim,
    constraint_dim,
):
    checkpoint_path = _resolve_sldac_checkpoint_path(args, example_name, run_tag, pretrain_episode, seed)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    algorithm = str(checkpoint.get("algorithm", ""))
    checkpoint_example = str(checkpoint.get("example_name", ""))
    checkpoint_run_tag = str(checkpoint.get("run_tag", ""))
    checkpoint_seed = int(checkpoint.get("seed", -1))
    if algorithm != "SLDAC":
        raise ValueError("checkpoint algorithm mismatch: expected SLDAC, got {0}".format(algorithm))
    if checkpoint_example != str(example_name):
        raise ValueError(
            "checkpoint example_name mismatch for run_tag={0}: expected {1}, got {2}".format(
                run_tag,
                example_name,
                checkpoint_example,
            )
        )
    if checkpoint_run_tag != str(run_tag):
        raise ValueError(
            "checkpoint run_tag mismatch: expected {0}, got {1}".format(run_tag, checkpoint_run_tag)
        )
    if checkpoint_seed != int(seed):
        raise ValueError(
            "checkpoint seed mismatch for run_tag={0}: expected {1}, got {2}".format(
                run_tag,
                int(seed),
                checkpoint_seed,
            )
        )

    shapes = checkpoint.get("shapes", {})
    expected_shapes = {
        "state_dim": int(state_dim),
        "action_dim": int(action_dim),
        "constraint_dim": int(constraint_dim),
    }
    for field_name, expected_value in expected_shapes.items():
        actual_value = int(shapes.get(field_name, -1))
        if actual_value != expected_value:
            raise ValueError(
                "checkpoint shape mismatch for {0}: expected {1}, got {2}".format(
                    field_name,
                    expected_value,
                    actual_value,
                )
            )

    model = checkpoint.get("model", {})
    actor_state_dict = model.get("actor_state_dict")
    actor_log_std = model.get("actor_log_std")
    if actor_state_dict is None:
        raise KeyError("checkpoint model.actor_state_dict is missing.")
    if actor_log_std is None:
        raise KeyError("checkpoint model.actor_log_std is missing.")

    if _is_mimo(example_name):
        actor = GaussianPolicy_MIMO(int(state_dim), int(action_dim), device, int(policy_batch_size))
    else:
        actor = GaussianPolicy_CLQR(int(state_dim), int(action_dim), device, int(policy_batch_size))
    actor.net.load_state_dict(actor_state_dict)
    actor.log_std = torch.as_tensor(actor_log_std, dtype=torch.float, device=device).detach().clone().view(-1)
    if int(actor.log_std.numel()) != int(action_dim):
        raise ValueError(
            "checkpoint actor_log_std size mismatch: expected {0}, got {1}".format(
                int(action_dim),
                int(actor.log_std.numel()),
            )
        )

    print("load old policy checkpoint:", checkpoint_path)
    return FrozenActorPolicy(actor)


def _train_sldac_like_actor(args, example_name, seed):
    _set_seed(seed)
    device = "cpu"
    env, actor, state_dim, action_dim, constraint_dim, constr_lim = _build_scene(example_name, seed, device, int(args.grad_T))

    t_horizon = int(args.T)
    grad_t = int(args.grad_T)
    num_new_data = int(args.num_new_data)
    update_time_per_episode = int(args.update_time_per_episode)
    max_steps = int(args.MAX_STEPS)
    alpha_pow = float(args.alpha_pow)
    beta_actor_pow = float(getattr(args, "beta_actor_pow", getattr(args, "beta_pow", 0.7)))
    beta_rho_pow = float(getattr(args, "beta_rho_pow", beta_actor_pow))
    xi0 = float(getattr(args, "xi0", DEFAULT_OFFLINE_WEIGHT))
    if beta_rho_pow <= beta_actor_pow:
        raise ValueError(
            "beta_rho_pow must be greater than beta_actor_pow. got beta_actor_pow={0}, beta_rho_pow={1}".format(
                beta_actor_pow,
                beta_rho_pow,
            )
        )
    if (xi0 < 0.0) or (xi0 > 1.0):
        raise ValueError("xi0 must be in [0, 1] as offline weight. got xi0={0}".format(xi0))
    eta_pow = float(args.eta_pow)
    gamma_pow_reward = float(args.gamma_pow_reward)
    gamma_pow_cost = float(args.gamma_pow_cost)
    tau_reward = float(args.tau_reward)
    tau_cost = float(args.tau_cost)
    q_update_time = int(args.Q_update_time)
    window = int(args.window)

    buffer = _make_buffer(example_name, t_horizon, window, num_new_data, state_dim, action_dim, constraint_dim)
    critic = Critic(example_name, grad_t, state_dim, action_dim, constraint_dim, q_update_time, device)

    theta = _flatten_actor(actor)
    func_value = np.zeros((constraint_dim + 1,), dtype=np.float64)
    grad = np.zeros((constraint_dim + 1, theta.size), dtype=np.float64)

    observation = env.reset()
    update_index = 0
    q_update_index = 0

    for t in range(max_steps):
        state = observation
        action = _sample_action(actor, state)
        observation, reward, done, info = env.step(action)
        next_state = observation
        costs = _build_costs(reward, info, constraint_dim, constr_lim)
        aver_cost = float(info.get("cost", 0.0)) / max(constraint_dim, 1)
        buffer.store_experiences(state, action, costs, next_state, float(reward), aver_cost)

        if t > 2 * t_horizon and ((t - 2 * t_horizon) % (num_new_data / q_update_time) == 0):
            q_update_index += 1
            alpha = 1.0 / ((update_index + 1) ** alpha_pow)
            beta_actor = 1.0 / ((update_index + 1) ** beta_actor_pow)
            beta_rho = 1.0 / ((update_index + 1) ** beta_rho_pow)
            eta = 1.0 / ((update_index + 1) ** eta_pow)
            if q_update_index == q_update_time:
                gamma_reward = 1.0 / ((update_index + 1) ** gamma_pow_reward)
                gamma_cost = 1.0 / ((update_index + 1) ** gamma_pow_cost)
            else:
                gamma_reward = 0.0
                gamma_cost = 0.0

            state_buffer, action_buffer, costs_buffer, next_state_buffer, _, _ = buffer.take_experiences()
            func_value_tilda = np.mean(costs_buffer, axis=0)
            func_value = (1.0 - alpha) * func_value + alpha * func_value_tilda

            state_batch = state_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            action_batch = action_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            costs_batch = costs_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            next_state_batch = next_state_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            next_action_batch = np.zeros((grad_t, action_dim), dtype=np.float64)
            for idx in range(grad_t):
                next_action_batch[idx, :] = _sample_action(actor, next_state_batch[idx, :])

            critic.critic_update(func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost)

            if q_update_index == q_update_time:
                update_index += 1
                q_update_index = 0
                state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=device)
                action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)
                q_hat_torch = critic.critic_value(state_batch_torch, action_batch_torch)
                q_hat = _q_head_normalize(q_hat_torch.detach().cpu().numpy())
                q_hat_torch = torch.tensor(q_hat, dtype=torch.float, device=device)

                grad_tilda = np.zeros_like(grad)
                for head in range(constraint_dim + 1):
                    _zero_actor_extra_grad(actor)
                    log_prob, log_std_leaf = _log_prob_batch(actor, state_batch_torch, action_batch_torch, require_grad=True)
                    actor_loss = (q_hat_torch[:, head] * log_prob).mean()
                    actor_loss.backward()
                    grad_tilda[head] = _flatten_actor_grad(actor, log_std_leaf.grad)
                grad = (1.0 - alpha) * grad + alpha * grad_tilda
                theta_bar = _policy_update(func_value, grad, theta, tau_reward=tau_reward, tau_cost=tau_cost)
                theta = (1.0 - beta) * theta + beta * theta_bar
                _set_actor_from_flat(actor, theta)

    return _clone_actor(actor)


def _build_old_policy_library(args, example_name, main_env, device, state_dim, action_dim, constraint_dim):
    seed = int(getattr(args, "seed", 0))
    old_policy_seed = int(getattr(args, "old_policy_seed", 0))
    old_policies = [_build_dk_policy(example_name, main_env, device, seed)]
    run_tags = _normalize_run_tags(getattr(args, "old_policy_run_tags", None))
    if not run_tags:
        print("old_policy_run_tags is empty: use DK-only library.")
        return old_policies
    pretrain_episode = int(getattr(args, "pretrain_episode", 0))
    if pretrain_episode <= 0:
        raise ValueError("pretrain_episode must be a positive integer.")
    for run_tag in run_tags:
        old_policies.append(
            _load_old_policy_from_checkpoint(
                args,
                example_name,
                run_tag,
                pretrain_episode,
                old_policy_seed,
                device,
                int(args.grad_T),
                state_dim,
                action_dim,
                constraint_dim,
            )
        )
    return old_policies


def _build_old_policy_labels(args):
    run_tags = _normalize_run_tags(getattr(args, "old_policy_run_tags", None))
    return ["new_actor", "dk_policy"] + run_tags


def _format_rho_debug_line(rho_labels, rho, precision=6):
    try:
        rho_arr = np.asarray(rho, dtype=np.float64).reshape(-1)
        labels = [] if rho_labels is None else [str(item) for item in list(rho_labels)]
        # 日志输出不能因为标签异常中断训练，维度不匹配时退化为索引标签。
        if len(labels) != rho_arr.size:
            labels = ["rho_{0}".format(idx) for idx in range(rho_arr.size)]
        value_format = "{{0}}={{1:.{0}f}}".format(int(precision))
        parts = []
        for idx, value in enumerate(rho_arr):
            numeric = float(value)
            if np.isfinite(numeric):
                parts.append(value_format.format(labels[idx], numeric))
            else:
                parts.append("{0}={1}".format(labels[idx], numeric))
        return "rho: " + ", ".join(parts)
    except Exception:
        rho_arr = np.asarray(rho).reshape(-1)
        return "rho: " + np.array2string(rho_arr, precision=int(precision), separator=", ")


def _select_policy_gradient_batch(
    online_state_batch,
    online_action_batch,
    offline_datasets,
    xi,
    grad_t,
    state_dim,
    action_dim,
    use_offline_data,
):
    # HRL 分支固定只使用在线样本；Fused-CPRO 保持在线/离线混合更新。
    return _build_policy_gradient_batch_impl(
        online_state_batch,
        online_action_batch,
        offline_datasets,
        xi,
        grad_t,
        state_dim,
        action_dim,
        use_offline_data,
    )


def _select_policy_gradient_batch_impl(
    online_state_batch,
    online_action_batch,
    offline_datasets,
    xi,
    grad_t,
    state_dim,
    action_dim,
    use_offline_data,
):
    # HRL keeps the policy-gradient batch purely online.
    return _build_policy_gradient_batch_impl(
        online_state_batch,
        online_action_batch,
        offline_datasets,
        xi,
        grad_t,
        state_dim,
        action_dim,
        use_offline_data,
    )


def _run_policy_mix_main(args, example_name, algorithm_label, use_offline_data, xi0):
    seed = int(getattr(args, "seed", 0))
    _set_seed(seed)
    device = str(getattr(args, "device", "cpu")).lower()
    if device == "cuda" and (not torch.cuda.is_available()):
        device = "cpu"

    t_horizon = int(args.T)
    grad_t = int(args.grad_T)
    num_new_data = int(args.num_new_data)
    update_time_per_episode = int(args.update_time_per_episode)
    max_steps = int(args.MAX_STEPS)
    alpha_pow = float(args.alpha_pow)
    beta_actor_pow = float(getattr(args, "beta_actor_pow", getattr(args, "beta_pow", 0.7)))
    rho_scheduler_config = _build_rho_scheduler_config(args, beta_actor_pow)
    xi0 = float(xi0)
    if (xi0 < 0.0) or (xi0 > 1.0):
        raise ValueError("xi0 must be in [0, 1] as offline weight. got xi0={0}".format(xi0))
    eta_pow = float(args.eta_pow)
    gamma_pow_reward = float(args.gamma_pow_reward)
    gamma_pow_cost = float(args.gamma_pow_cost)
    tau_reward = float(args.tau_reward)
    tau_cost = float(args.tau_cost)
    q_update_time = int(args.Q_update_time)
    window = int(args.window)

    env, actor_new, state_dim, action_dim, constraint_dim, constr_lim = _build_scene(example_name, seed, device, grad_t)
    observation = env.reset()

    old_policies = _build_old_policy_library(
        args,
        example_name,
        env,
        device,
        state_dim,
        action_dim,
        constraint_dim,
    )
    rho_labels = _build_old_policy_labels(args)
    offline_datasets = []
    if use_offline_data:
        offline_steps = max(int(OFFLINE_STEPS_MULTIPLIER * t_horizon), grad_t)
        for idx, policy in enumerate(old_policies):
            offline_datasets.append(_policy_rollout_dataset(example_name, policy, offline_steps, seed + 1000 + idx, device))

    buffer = _make_buffer(example_name, t_horizon, window, num_new_data, state_dim, action_dim, constraint_dim)
    critic = Critic(example_name, grad_t, state_dim, action_dim, constraint_dim, q_update_time, device)

    rho_lower_bounds = _resolve_rho_lower_bounds(args, len(old_policies) + 1)
    rho = np.ones((len(old_policies) + 1,), dtype=np.float64)
    rho = _normalize_simplex(rho, rho_lower_bounds)
    theta_actor = _flatten_actor(actor_new)
    theta_dim = rho.size + theta_actor.size
    func_value = np.zeros((constraint_dim + 1,), dtype=np.float64)
    grad = np.zeros((constraint_dim + 1, theta_dim), dtype=np.float64)

    reward_average_save = []
    cost_average_save = []
    rho_history_save = []
    xi_history_save = []
    drift_update_index_save = []
    actor_drift_save = []
    critic_drift_save = []
    rho_drift_save = []
    update_index = 0
    print_index = 0
    q_update_index = 0
    critic_anchor = critic.flatten_parameters(include_target=True).copy()

    for t in range(max_steps):
        state = observation
        choice = int(np.random.choice(rho.size, p=rho))
        if choice == 0:
            action = _sample_action(actor_new, state)
        else:
            action = old_policies[choice - 1].sample_action(state)
        observation, reward, done, info = env.step(action)
        next_state = observation
        costs = _build_costs(reward, info, constraint_dim, constr_lim)
        aver_cost = float(info.get("cost", 0.0)) / max(constraint_dim, 1)
        buffer.store_experiences(state, action, costs, next_state, float(reward), aver_cost)

        if t > 2 * t_horizon and ((t - 2 * t_horizon) % (num_new_data / q_update_time) == 0):
            q_update_index += 1
            alpha = 1.0 / ((update_index + 1) ** alpha_pow)
            beta_actor = 1.0 / ((update_index + 1) ** beta_actor_pow)
            beta_rho = _get_rho_beta(update_index, rho_scheduler_config)
            if use_offline_data:
                xi = _get_xi(update_index, rho_scheduler_config["xi_decay_pow"], xi0)
            else:
                xi = 0.0
            eta = 1.0 / ((update_index + 1) ** eta_pow)
            if q_update_index == q_update_time:
                gamma_reward = 1.0 / ((update_index + 1) ** gamma_pow_reward)
                gamma_cost = 1.0 / ((update_index + 1) ** gamma_pow_cost)
            else:
                gamma_reward = 0.0
                gamma_cost = 0.0

            state_buffer, action_buffer, costs_buffer, next_state_buffer, aver_reward_buffer, aver_cost_buffer = buffer.take_experiences()
            func_value_tilda = np.mean(costs_buffer, axis=0)
            func_value = (1.0 - alpha) * func_value + alpha * func_value_tilda
            if (update_index % update_time_per_episode == 0) and (q_update_index == 1):
                print("{0}_EPISODE:".format(algorithm_label), print_index)
                print("reward_average:", float(np.mean(aver_reward_buffer)))
                print("cost_average:", float(np.mean(aver_cost_buffer)))
                reward_average_save.append(float(np.mean(aver_reward_buffer)))
                cost_average_save.append(float(np.mean(aver_cost_buffer)))
                rho_history_save.append(np.asarray(rho, dtype=np.float64).copy())
                print(_format_rho_debug_line(rho_labels, rho))
                if use_offline_data:
                    xi_history_save.append(float(xi))
                    print("xi_offline_weight:", float(xi))
                print_index += 1

            online_state_batch = state_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            online_action_batch = action_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            online_costs_batch = costs_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            next_state_batch = next_state_buffer[(2 * t_horizon - grad_t):2 * t_horizon]
            next_action_batch = np.zeros((grad_t, action_dim), dtype=np.float64)
            for idx in range(grad_t):
                next_action_batch[idx, :] = _sample_action(actor_new, next_state_batch[idx, :])

            critic.critic_update(func_value, online_state_batch, online_action_batch, online_costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost)

            if q_update_index == q_update_time:
                update_index += 1
                q_update_index = 0
                critic_now = critic.flatten_parameters(include_target=True).copy()
                critic_drift = _rms_drift(critic_now, critic_anchor)

                fused_state_batch, fused_action_batch = _select_policy_gradient_batch_impl(
                    online_state_batch,
                    online_action_batch,
                    offline_datasets,
                    xi,
                    grad_t,
                    state_dim,
                    action_dim,
                    use_offline_data,
                )

                fused_state_torch = torch.tensor(fused_state_batch, dtype=torch.float, device=device)
                fused_action_torch = torch.tensor(fused_action_batch, dtype=torch.float, device=device)
                q_hat_torch = critic.critic_value(fused_state_torch, fused_action_torch)
                q_hat = _q_head_normalize(q_hat_torch.detach().cpu().numpy())
                q_hat_torch = torch.tensor(q_hat, dtype=torch.float, device=device)
                online_batch_size = grad_t
                online_state_torch = fused_state_torch[:online_batch_size]
                online_action_torch = fused_action_torch[:online_batch_size]
                q_hat_online_torch = q_hat_torch[:online_batch_size]
                offline_state_torch = fused_state_torch[online_batch_size:]
                offline_action_torch = fused_action_torch[online_batch_size:]
                q_hat_offline_torch = q_hat_torch[online_batch_size:]

                grad_tilda = np.zeros_like(grad)
                for head in range(constraint_dim + 1):
                    _zero_actor_extra_grad(actor_new)
                    rho_torch = torch.tensor(rho, dtype=torch.float, device=device, requires_grad=True)
                    log_mix_online, rho_torch, log_std_online = _build_mixture_log_prob(
                        online_state_torch,
                        online_action_torch,
                        actor_new,
                        old_policies,
                        rho,
                        rho_lower_bounds,
                        rho_torch=rho_torch,
                    )
                    actor_loss_online = (q_hat_online_torch[:, head] * log_mix_online).mean()
                    log_std_offline = None
                    actor_loss_offline = None
                    if use_offline_data and q_hat_offline_torch.shape[0] > 0:
                        log_mix_offline, rho_torch, log_std_offline = _build_mixture_log_prob(
                            offline_state_torch,
                            offline_action_torch,
                            actor_new,
                            old_policies,
                            rho,
                            rho_lower_bounds,
                            rho_torch=rho_torch,
                        )
                        actor_loss_offline = (q_hat_offline_torch[:, head] * log_mix_offline).mean()
                    actor_loss = _blend_online_offline_loss(actor_loss_online, actor_loss_offline, xi, use_offline_data)
                    actor_loss.backward()
                    rho_grad = rho_torch.grad.detach().cpu().numpy().astype(np.float64, copy=False)
                    actor_grad = _flatten_actor_grad(
                        actor_new,
                        _merge_log_std_grads(
                            log_std_online.grad,
                            None if log_std_offline is None else log_std_offline.grad,
                        ),
                    )
                    grad_tilda[head] = np.concatenate((rho_grad, actor_grad), axis=0)

                grad = (1.0 - alpha) * grad + alpha * grad_tilda
                actor_now = _flatten_actor(actor_new)
                theta_now = np.concatenate((rho, actor_now), axis=0)
                theta_bar = _policy_update(
                    func_value,
                    grad,
                    theta_now,
                    tau_reward=tau_reward,
                    tau_cost=tau_cost,
                    simplex_dim=rho.size,
                    rho_lower_bounds=rho_lower_bounds,
                )
                rho_bar = theta_bar[:rho.size]
                actor_bar = theta_bar[rho.size:]
                rho_next = (1.0 - beta_rho) * rho + beta_rho * rho_bar
                actor_next = (1.0 - beta_actor) * actor_now + beta_actor * actor_bar
                rho_applied = _normalize_simplex(rho_next, rho_lower_bounds)
                actor_drift_save.append(_rms_drift(actor_next, actor_now))
                critic_drift_save.append(critic_drift)
                rho_drift_save.append(_rms_drift(rho_applied, rho))
                drift_update_index_save.append(int(update_index))
                critic_anchor = critic_now
                rho = rho_applied
                _set_actor_from_flat(actor_new, actor_next)

    drift_history = {
        "update_index": np.asarray(drift_update_index_save, dtype=np.int32),
        "actor_rms": np.asarray(actor_drift_save, dtype=np.float64),
        "critic_rms": np.asarray(critic_drift_save, dtype=np.float64),
        "rho_rms": np.asarray(rho_drift_save, dtype=np.float64),
    }

    return (
        reward_average_save,
        cost_average_save,
        np.asarray(rho_history_save, dtype=np.float64),
        np.asarray(xi_history_save, dtype=np.float64),
        rho_labels,
        drift_history,
    )


def _run_prcrl_main(args, example_name):
    seed = int(getattr(args, "seed", 0))
    _set_seed(seed)
    device = str(getattr(args, "device", "cpu")).lower()
    if device == "cuda" and (not torch.cuda.is_available()):
        device = "cpu"

    t_horizon = int(args.T)
    grad_t = int(args.grad_T)
    if grad_t != t_horizon:
        raise ValueError(
            "PRCRL requires grad_T == T in the current implementation. got T={0}, grad_T={1}".format(
                t_horizon,
                grad_t,
            )
        )
    num_new_data = int(args.num_new_data)
    if num_new_data <= 0:
        raise ValueError("num_new_data must be positive for PRCRL.")
    update_time_per_episode = int(args.update_time_per_episode)
    max_steps = int(args.MAX_STEPS)
    alpha_pow = float(args.alpha_pow)
    beta_actor_pow = float(getattr(args, "beta_actor_pow", getattr(args, "beta_pow", 0.7)))
    rho_scheduler_config = _build_rho_scheduler_config(args, beta_actor_pow)
    tau_reward = float(args.tau_reward)
    tau_cost = float(args.tau_cost)
    window = int(args.window)

    env, actor_new, state_dim, action_dim, constraint_dim, constr_lim = _build_scene(example_name, seed, device, grad_t)
    observation = env.reset()
    old_policies = _build_old_policy_library(
        args,
        example_name,
        env,
        device,
        state_dim,
        action_dim,
        constraint_dim,
    )
    rho_labels = _build_old_policy_labels(args)
    buffer = _make_buffer(example_name, t_horizon, window, num_new_data, state_dim, action_dim, constraint_dim)

    rho_lower_bounds = _resolve_rho_lower_bounds(args, len(old_policies) + 1)
    rho = np.ones((len(old_policies) + 1,), dtype=np.float64)
    rho = _normalize_simplex(rho, rho_lower_bounds)
    theta_actor = _flatten_actor(actor_new)
    theta_dim = rho.size + theta_actor.size
    func_value = np.zeros((constraint_dim + 1,), dtype=np.float64)
    grad = np.zeros((constraint_dim + 1, theta_dim), dtype=np.float64)

    reward_average_save = []
    cost_average_save = []
    rho_history_save = []
    drift_update_index_save = []
    actor_drift_save = []
    rho_drift_save = []
    update_index = 0
    print_index = 0

    for t in range(max_steps):
        state = observation
        choice = int(np.random.choice(rho.size, p=rho))
        if choice == 0:
            action = _sample_action(actor_new, state)
        else:
            action = old_policies[choice - 1].sample_action(state)
        observation, reward, done, info = env.step(action)
        next_state = observation
        costs = _build_costs(reward, info, constraint_dim, constr_lim)
        aver_cost = float(info.get("cost", 0.0)) / max(constraint_dim, 1)
        buffer.store_experiences(state, action, costs, next_state, float(reward), aver_cost)

        if (t + 1) < (2 * t_horizon):
            continue
        if ((t + 1 - 2 * t_horizon) % num_new_data) != 0:
            continue

        alpha = 1.0 / ((update_index + 1) ** alpha_pow)
        beta_actor = 1.0 / ((update_index + 1) ** beta_actor_pow)
        beta_rho = _get_rho_beta(update_index, rho_scheduler_config)

        state_buffer, action_buffer, costs_buffer, next_state_buffer, aver_reward_buffer, aver_cost_buffer = buffer.take_experiences()
        func_value_tilda = np.mean(costs_buffer, axis=0)
        func_value = (1.0 - alpha) * func_value + alpha * func_value_tilda

        if update_index % update_time_per_episode == 0:
            print("PRCRL_EPISODE:", print_index)
            print("reward_average:", float(np.mean(aver_reward_buffer)))
            print("cost_average:", float(np.mean(aver_cost_buffer)))
            reward_average_save.append(float(np.mean(aver_reward_buffer)))
            cost_average_save.append(float(np.mean(aver_cost_buffer)))
            rho_history_save.append(np.asarray(rho, dtype=np.float64).copy())
            print(_format_rho_debug_line(rho_labels, rho))
            print_index += 1

        q_hat = _prcrl_window_q_hat(costs_buffer, func_value, t_horizon)
        q_hat = _prcrl_q_head_normalize(q_hat)
        q_hat_torch = torch.tensor(q_hat, dtype=torch.float, device=device)
        state_batch_torch = torch.tensor(state_buffer[1 : t_horizon + 1], dtype=torch.float, device=device)
        action_batch_torch = torch.tensor(action_buffer[1 : t_horizon + 1], dtype=torch.float, device=device)

        grad_tilda = np.zeros_like(grad)
        for head in range(constraint_dim + 1):
            _zero_actor_extra_grad(actor_new)
            log_mix, rho_torch, log_std_leaf = _build_mixture_log_prob(
                state_batch_torch,
                action_batch_torch,
                actor_new,
                old_policies,
                rho,
                rho_lower_bounds,
            )
            actor_loss = (q_hat_torch[:, head] * log_mix).mean()
            actor_loss.backward()
            rho_grad = rho_torch.grad.detach().cpu().numpy().astype(np.float64, copy=False)
            actor_grad = _flatten_actor_grad(actor_new, log_std_leaf.grad)
            grad_tilda[head] = np.concatenate((rho_grad, actor_grad), axis=0)

        grad = (1.0 - alpha) * grad + alpha * grad_tilda
        actor_now = _flatten_actor(actor_new)
        theta_now = np.concatenate((rho, actor_now), axis=0)
        theta_bar = _policy_update(
            func_value,
            grad,
            theta_now,
            tau_reward=tau_reward,
            tau_cost=tau_cost,
            simplex_dim=rho.size,
            rho_lower_bounds=rho_lower_bounds,
        )
        rho_bar = theta_bar[:rho.size]
        actor_bar = theta_bar[rho.size:]
        rho_next = (1.0 - beta_rho) * rho + beta_rho * rho_bar
        actor_next = (1.0 - beta_actor) * actor_now + beta_actor * actor_bar
        rho_applied = _normalize_simplex(rho_next, rho_lower_bounds)

        update_index += 1
        drift_update_index_save.append(int(update_index))
        actor_drift_save.append(_rms_drift(actor_next, actor_now))
        rho_drift_save.append(_rms_drift(rho_applied, rho))
        rho = rho_applied
        _set_actor_from_flat(actor_new, actor_next)

    rho_history = np.asarray(rho_history_save, dtype=np.float64)
    if rho_history.size <= 0:
        rho_history = np.zeros((0, len(rho_labels)), dtype=np.float64)
    elif rho_history.ndim == 1:
        rho_history = rho_history.reshape(1, -1)

    drift_history = {
        "update_index": np.asarray(drift_update_index_save, dtype=np.int32),
        "actor_rms": np.asarray(actor_drift_save, dtype=np.float64),
        "critic_rms": np.zeros((0,), dtype=np.float64),
        "rho_rms": np.asarray(rho_drift_save, dtype=np.float64),
    }
    return (
        reward_average_save,
        cost_average_save,
        rho_history,
        np.zeros((0,), dtype=np.float64),
        rho_labels,
        drift_history,
    )


def Fused_CPRO_main(args, example_name):
    xi0 = float(getattr(args, "xi0", DEFAULT_OFFLINE_WEIGHT))
    xi_decay_updates = int(getattr(args, "xi_decay_updates", 500))
    if xi_decay_updates <= 0:
        raise ValueError("xi_decay_updates must be a positive integer. got xi_decay_updates={0}".format(xi_decay_updates))
    return _run_policy_mix_main(
        args,
        example_name,
        algorithm_label="Fused_CPRO",
        use_offline_data=True,
        xi0=xi0,
    )


def Fused_CPRO_CosRho_main(args, example_name):
    run_args = copy.copy(args)
    run_args.rho_scheduler = getattr(run_args, "rho_scheduler", RHO_SCHEDULER_COSINE_RESTART_DECAY)
    xi0 = float(getattr(run_args, "xi0", DEFAULT_OFFLINE_WEIGHT))
    return _run_policy_mix_main(
        run_args,
        example_name,
        algorithm_label="Fused_CPRO_CosRho",
        use_offline_data=True,
        xi0=xi0,
    )


def HRL_main(args, example_name):
    return _run_policy_mix_main(
        args,
        example_name,
        algorithm_label="HRL",
        use_offline_data=False,
        xi0=0.0,
    )


def PRCRL_main(args, example_name):
    return _run_prcrl_main(args, example_name)
