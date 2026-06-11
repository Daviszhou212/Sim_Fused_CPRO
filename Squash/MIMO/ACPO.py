import copy
import math

import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize_scalar

from environment import Environment_CLQR, Environment_MIMO
from model import GaussianPolicy_CLQR, GaussianPolicy_MIMO
from seed_utils import resolve_torch_device


# 数值稳定项，统一用于归一化、开方与除法。
EPS = 1e-8
# 状态标准化裁剪范围，避免早期统计量不稳定时放大输入。
STATE_NORM_CLIP = 10.0
# CG 迭代残差阈值，控制 Fisher 线性系统求解精度。
CG_RESIDUAL_TOL = 1e-10
# line search 判定时允许的极小松弛量。
LINE_SEARCH_TOL = 1e-8

# MIMO 默认场景配置：与当前 MIMO2 主线实验保持一致。
MIMO_DEFAULT_NT = 8
MIMO_DEFAULT_UE_NUM = 4
MIMO_DEFAULT_CONSTRAINT_LIMIT = 1.2

# CLQR 默认场景配置：与当前 CLQR ACPO 入口保持一致。
CLQR_DEFAULT_STATE_DIM = 15
CLQR_DEFAULT_ACTION_DIM = 4
CLQR_DEFAULT_CONSTRAINT_LIMIT = 380.0


def _scene_name(example_name):
    text = str(example_name).strip()
    if "MIMO" in text:
        return "MIMO"
    if text == "CLQR":
        return "CLQR"
    raise ValueError("ACPO_main only supports MIMO or CLQR. got example_name={0}".format(example_name))


def _is_mimo(example_name):
    return _scene_name(example_name) == "MIMO"


def _default_constraint_limit(example_name):
    if _is_mimo(example_name):
        return float(MIMO_DEFAULT_CONSTRAINT_LIMIT)
    return float(CLQR_DEFAULT_CONSTRAINT_LIMIT)


class RunningNormalizer:
    """运行时状态归一化器。"""

    def __init__(self, dim, clip=STATE_NORM_CLIP):
        self.dim = int(dim)
        self.clip = float(clip)
        self.count = 0
        self.mean = np.zeros((self.dim,), dtype=np.float64)
        self.m2 = np.zeros((self.dim,), dtype=np.float64)

    def update(self, batch):
        values = np.asarray(batch, dtype=np.float64).reshape(-1, self.dim)
        for row in values:
            self.count += 1
            delta = row - self.mean
            self.mean = self.mean + delta / float(self.count)
            delta2 = row - self.mean
            self.m2 = self.m2 + delta * delta2

    def normalize(self, values):
        arr = np.asarray(values, dtype=np.float64)
        if self.count <= 1:
            return arr.copy()
        variance = self.m2 / float(max(self.count - 1, 1))
        std = np.sqrt(np.maximum(variance, EPS))
        normalized = (arr - self.mean) / std
        return np.clip(normalized, -self.clip, self.clip)


class ValueNet(nn.Module):
    """ACPO 用于平均奖励 bias 的状态值网络。"""

    def __init__(self, state_dim):
        super(ValueNet, self).__init__()
        self.fc1 = nn.Linear(int(state_dim), 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 1)
        nn.init.orthogonal_(self.fc1.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias, 0.0)
        nn.init.orthogonal_(self.fc2.weight, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias, 0.0)
        nn.init.orthogonal_(self.fc3.weight, gain=1.0)
        nn.init.constant_(self.fc3.bias, 0.0)

    def forward(self, state_torch):
        x = torch.tanh(self.fc1(state_torch))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x).squeeze(-1)


class AverageValueCritic:
    """平均奖励 / 平均代价的 bias critic。"""

    def __init__(self, state_dim, device, learning_rate):
        self.device = device
        self.net = ValueNet(state_dim).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=float(learning_rate))

    def predict(self, states_torch):
        with torch.no_grad():
            return self.net(states_torch)

    def fit(self, states_np, targets_np, epochs, batch_size):
        states = torch.tensor(states_np, dtype=torch.float32, device=self.device)
        targets = torch.tensor(targets_np, dtype=torch.float32, device=self.device)
        num_samples = int(states.shape[0])
        if num_samples <= 0:
            return
        mini_batch = min(int(batch_size), num_samples)
        for _ in range(int(epochs)):
            permutation = torch.randperm(num_samples, device=self.device)
            for start in range(0, num_samples, mini_batch):
                indices = permutation[start : start + mini_batch]
                pred = self.net(states[indices])
                loss = torch.mean((pred - targets[indices]) ** 2)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()


def _set_seed(seed):
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))


def _create_actor(example_name, state_dim, action_dim, device, batch_size, init_log_std):
    if _is_mimo(example_name):
        actor = GaussianPolicy_MIMO(int(state_dim), int(action_dim), device, int(batch_size))
    else:
        actor = GaussianPolicy_CLQR(int(state_dim), int(action_dim), device, int(batch_size))
    actor.log_std = torch.full(
        (int(action_dim),),
        float(init_log_std),
        dtype=torch.float32,
        device=device,
        requires_grad=True,
    )
    return actor


def _build_scene(example_name, seed, device, policy_batch_size, init_log_std, args):
    if _is_mimo(example_name):
        nt = int(getattr(args, "Nt", MIMO_DEFAULT_NT))
        ue_num = int(getattr(args, "UE_num", MIMO_DEFAULT_UE_NUM))
        state_dim = int(2 * ue_num * nt + ue_num)
        action_dim = int(ue_num + 1)
        env = Environment_MIMO(seed=int(seed), Nt=nt, UE_num=ue_num)
        cost_scale = float(max(ue_num, 1))
    else:
        state_dim = int(getattr(args, "state_dim", CLQR_DEFAULT_STATE_DIM))
        action_dim = int(getattr(args, "action_dim", CLQR_DEFAULT_ACTION_DIM))
        env = Environment_CLQR(seed=int(seed), state_dim=state_dim, action_dim=action_dim)
        cost_scale = 1.0
    actor = _create_actor(example_name, state_dim, action_dim, device, policy_batch_size, init_log_std)
    return env, actor, state_dim, action_dim, cost_scale


def _clone_actor(actor):
    cloned = copy.deepcopy(actor)
    cloned.log_std = actor.log_std.detach().clone().to(actor.device)
    cloned.log_std.requires_grad_(False)
    for para in cloned.net.parameters():
        para.requires_grad_(False)
    return cloned


def _actor_tensors(actor):
    return list(actor.net.parameters()) + [actor.log_std]


def _flatten_tensors(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors], dim=0)


def _get_flat_params(actor):
    params = [para.detach() for para in actor.net.parameters()]
    params.append(actor.log_std.detach())
    return _flatten_tensors(params)


def _set_flat_params(actor, flat_params):
    vector = flat_params.detach().to(actor.device)
    offset = 0
    with torch.no_grad():
        for para in actor.net.parameters():
            numel = para.numel()
            para.copy_(vector[offset : offset + numel].view_as(para))
            offset += numel
    actor.log_std = vector[offset : offset + actor.action_dim].detach().clone()
    actor.log_std.requires_grad_(True)


def _dist_stats(actor, states_torch):
    mean = actor.net(states_torch)
    log_std = actor.log_std.view(1, -1).expand_as(mean)
    std = torch.exp(log_std)
    return mean, log_std, std


def _log_prob(actor, states_torch, actions_torch):
    mean, log_std, std = _dist_stats(actor, states_torch)
    dist = torch.distributions.Normal(mean, std)
    return dist.log_prob(actions_torch).sum(dim=1)


def _mean_kl(old_actor, new_actor, states_torch):
    old_mean, old_log_std, old_std = _dist_stats(old_actor, states_torch)
    new_mean, new_log_std, new_std = _dist_stats(new_actor, states_torch)
    numerator = old_std.pow(2) + (old_mean - new_mean).pow(2)
    denominator = 2.0 * new_std.pow(2) + EPS
    kl = (new_log_std - old_log_std) + numerator / denominator - 0.5
    return kl.sum(dim=1).mean()


def _flat_grad(loss, tensors, retain_graph=False, create_graph=False):
    grads = torch.autograd.grad(
        loss,
        tensors,
        retain_graph=retain_graph,
        create_graph=create_graph,
        allow_unused=False,
    )
    return _flatten_tensors(grads)


def _surrogate_grad(actor, states_torch, actions_torch, advantages_torch):
    log_prob = _log_prob(actor, states_torch, actions_torch)
    surrogate = torch.mean(log_prob * advantages_torch)
    return _flat_grad(surrogate, _actor_tensors(actor))


def _fisher_vector_product(actor, old_actor, states_torch, vector_torch, damping):
    tensors = _actor_tensors(actor)
    mean_kl = _mean_kl(old_actor, actor, states_torch)
    grad_kl = _flat_grad(mean_kl, tensors, retain_graph=True, create_graph=True)
    kl_vector = torch.dot(grad_kl, vector_torch)
    hvp = _flat_grad(kl_vector, tensors, retain_graph=False, create_graph=False)
    return hvp + float(damping) * vector_torch


def _conjugate_gradient(matvec, b_torch, max_iter, tol):
    x = torch.zeros_like(b_torch)
    r = b_torch.clone()
    p = r.clone()
    rr_old = torch.dot(r, r)
    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        if rr_old.item() <= float(tol):
            break
        ap = matvec(p)
        alpha = rr_old / (torch.dot(p, ap) + EPS)
        x = x + alpha * p
        r = r - alpha * ap
        rr_new = torch.dot(r, r)
        if rr_new.item() <= float(tol):
            rr_old = rr_new
            break
        beta = rr_new / (rr_old + EPS)
        p = r + beta * p
        rr_old = rr_new
    return x, iterations


def _normalize_advantages(advantages):
    arr = np.asarray(advantages, dtype=np.float64).reshape(-1)
    return (arr - np.mean(arr)) / (np.std(arr) + 1e-6)


def _discount_cumsum(deltas, lam):
    arr = np.asarray(deltas, dtype=np.float64).reshape(-1)
    out = np.zeros_like(arr)
    running = 0.0
    for idx in range(arr.size - 1, -1, -1):
        running = arr[idx] + float(lam) * running
        out[idx] = running
    return out


def _rollout_batch(env, actor, normalizer, horizon, constraint_limit, device, observation, cost_scale):
    states = []
    norm_states = []
    actions = []
    objective_costs = []
    internal_rewards = []
    costs = []
    cost_residuals = []
    next_states = []
    norm_next_states = []
    for _ in range(int(horizon)):
        state = np.asarray(observation, dtype=np.float64).reshape(-1)
        state_norm = normalizer.normalize(state)
        state_torch = torch.tensor(state_norm, dtype=torch.float32, device=device)
        with torch.no_grad():
            mean, _, std = _dist_stats(actor, state_torch.view(1, -1))
            dist = torch.distributions.Normal(mean, std)
            action = dist.sample().view(-1).detach().cpu().numpy()

        observation, reward, done, info = env.step(action)
        next_state = np.asarray(observation, dtype=np.float64).reshape(-1)
        next_state_norm = normalizer.normalize(next_state)

        objective_cost = float(reward)
        constraint_cost = float(info.get("cost", 0.0)) / float(max(cost_scale, 1.0))

        states.append(state)
        norm_states.append(state_norm)
        actions.append(action)
        objective_costs.append(objective_cost)
        internal_rewards.append(-objective_cost)
        costs.append(constraint_cost)
        cost_residuals.append(constraint_cost - float(constraint_limit))
        next_states.append(next_state)
        norm_next_states.append(next_state_norm)

        if done:
            observation = env.reset()

    states_np = np.asarray(states, dtype=np.float64)
    next_states_np = np.asarray(next_states, dtype=np.float64)
    normalizer.update(np.concatenate((states_np, next_states_np), axis=0))
    return {
        "state": states_np,
        "state_norm": np.asarray(norm_states, dtype=np.float64),
        "action": np.asarray(actions, dtype=np.float64),
        "objective_cost": np.asarray(objective_costs, dtype=np.float64),
        "reward_internal": np.asarray(internal_rewards, dtype=np.float64),
        "cost": np.asarray(costs, dtype=np.float64),
        "cost_residual": np.asarray(cost_residuals, dtype=np.float64),
        "next_state": next_states_np,
        "next_state_norm": np.asarray(norm_next_states, dtype=np.float64),
        "last_observation": np.asarray(observation, dtype=np.float64).reshape(-1),
    }


def _prepare_advantages(batch, reward_critic, cost_critic, lam_reward, lam_cost, device):
    state_torch = torch.tensor(batch["state_norm"], dtype=torch.float32, device=device)
    next_state_torch = torch.tensor(batch["next_state_norm"], dtype=torch.float32, device=device)
    reward_values = reward_critic.predict(state_torch).cpu().numpy()
    reward_next_values = reward_critic.predict(next_state_torch).cpu().numpy()
    cost_values = cost_critic.predict(state_torch).cpu().numpy()
    cost_next_values = cost_critic.predict(next_state_torch).cpu().numpy()

    reward_rate = float(np.mean(batch["reward_internal"]))
    cost_rate = float(np.mean(batch["cost"]))

    reward_deltas = batch["reward_internal"] - reward_rate + reward_next_values - reward_values
    cost_deltas = batch["cost"] - cost_rate + cost_next_values - cost_values

    reward_adv_raw = _discount_cumsum(reward_deltas, lam_reward)
    cost_adv_raw = _discount_cumsum(cost_deltas, lam_cost)
    reward_returns = reward_adv_raw + reward_values
    cost_returns = cost_adv_raw + cost_values

    return {
        "reward_rate": reward_rate,
        "cost_rate": cost_rate,
        "reward_adv_raw": reward_adv_raw,
        "cost_adv_raw": cost_adv_raw,
        "reward_adv_norm": _normalize_advantages(reward_adv_raw),
        "cost_adv_norm": _normalize_advantages(cost_adv_raw),
        "reward_returns": reward_returns,
        "cost_returns": cost_returns,
    }


def _solve_feasible_step(g_torch, a_torch, hvp_fn, delta, cg_iters):
    h_inv_g, cg_g = _conjugate_gradient(hvp_fn, g_torch, cg_iters, CG_RESIDUAL_TOL)
    h_inv_a, cg_a = _conjugate_gradient(hvp_fn, a_torch, cg_iters, CG_RESIDUAL_TOL)
    q = float(torch.dot(g_torch, h_inv_g).item())
    r = float(torch.dot(g_torch, h_inv_a).item())
    s = float(torch.dot(a_torch, h_inv_a).item())
    if q <= 0.0:
        return None, max(cg_g, cg_a)
    trpo_scale = math.sqrt(max(2.0 * float(delta) / (q + EPS), 0.0))
    trpo_step = trpo_scale * h_inv_g
    return {
        "a_torch": a_torch,
        "h_inv_g": h_inv_g,
        "h_inv_a": h_inv_a,
        "q": q,
        "r": r,
        "s": s,
        "trpo_step": trpo_step,
    }, max(cg_g, cg_a)


def _dual_objective(mu_value, q_value, r_value, s_value, c_value, delta_value):
    quad = q_value - 2.0 * mu_value * r_value + (mu_value ** 2) * s_value
    if quad <= 0.0:
        return -np.inf
    return mu_value * c_value - math.sqrt(max(2.0 * delta_value * quad, EPS))


def _primal_feasible_step(step_terms, c_value, delta_value, hvp_fn):
    a_torch = step_terms["a_torch"]
    h_inv_g = step_terms["h_inv_g"]
    h_inv_a = step_terms["h_inv_a"]
    q_value = float(step_terms["q"])
    r_value = float(step_terms["r"])
    s_value = float(step_terms["s"])

    trpo_step = step_terms["trpo_step"]
    if c_value + float(torch.dot(a_torch, trpo_step).item()) <= 0.0:
        return trpo_step, 0.0

    mu_upper = max(1.0, (2.0 * abs(r_value) / (s_value + EPS)) + abs(c_value) * 10.0 + 1.0)
    opt_result = minimize_scalar(
        lambda mu_value: -_dual_objective(mu_value, q_value, r_value, s_value, c_value, float(delta_value)),
        bounds=(0.0, float(mu_upper)),
        method="bounded",
        options={"xatol": 1e-6},
    )
    mu_value = float(opt_result.x) if opt_result.success else 0.0
    quad = q_value - 2.0 * mu_value * r_value + (mu_value ** 2) * s_value
    if quad <= 0.0:
        return None, mu_value
    lam_value = math.sqrt(max(quad / max(2.0 * float(delta_value), EPS), EPS))
    step = (h_inv_g - mu_value * h_inv_a) / (lam_value + EPS)
    quad_value = float(torch.dot(step, hvp_fn(step)).item())
    if quad_value > 2.0 * float(delta_value):
        step = step * math.sqrt(max((2.0 * float(delta_value)) / (quad_value + EPS), 0.0))
    return step, mu_value


def _recovery_step(g_torch, a_torch, hvp_fn, delta, recovery_t, cg_iters):
    h_inv_g, cg_g = _conjugate_gradient(hvp_fn, g_torch, cg_iters, CG_RESIDUAL_TOL)
    h_inv_a, cg_a = _conjugate_gradient(hvp_fn, a_torch, cg_iters, CG_RESIDUAL_TOL)
    reward_norm = math.sqrt(max(float(torch.dot(g_torch, h_inv_g).item()), EPS))
    cost_norm = math.sqrt(max(float(torch.dot(a_torch, h_inv_a).item()), EPS))
    mixed = float(recovery_t) * (h_inv_a / cost_norm) + (1.0 - float(recovery_t)) * (h_inv_g / reward_norm)
    raw_step = -math.sqrt(max(2.0 * float(delta), 0.0)) * mixed
    quad_value = float(torch.dot(raw_step, hvp_fn(raw_step)).item())
    if quad_value > 2.0 * float(delta):
        raw_step = raw_step * math.sqrt(max((2.0 * float(delta)) / (quad_value + EPS), 0.0))
    return raw_step, max(cg_g, cg_a)


def _evaluate_candidate(
    actor,
    old_actor,
    states_torch,
    actions_torch,
    old_log_prob_torch,
    reward_adv_raw_torch,
    cost_adv_raw_torch,
    old_cost_mean,
):
    new_log_prob = _log_prob(actor, states_torch, actions_torch)
    ratio = torch.exp(new_log_prob - old_log_prob_torch)
    reward_surr = torch.mean(ratio * reward_adv_raw_torch).item()
    cost_surr = float(old_cost_mean + torch.mean(ratio * cost_adv_raw_torch).item())
    mean_kl = float(_mean_kl(old_actor, actor, states_torch).item())
    return reward_surr, cost_surr, mean_kl


def _line_search(
    actor,
    old_actor,
    old_params,
    step_direction,
    states_torch,
    actions_torch,
    old_log_prob_torch,
    reward_adv_raw_torch,
    cost_adv_raw_torch,
    old_cost_mean,
    constraint_limit,
    delta,
    backtrack_coeff,
    max_backtracks,
    mode,
):
    step_vector = step_direction.detach().clone()
    for backtrack in range(int(max_backtracks) + 1):
        scale = float(backtrack_coeff) ** backtrack
        candidate = old_params + scale * step_vector
        _set_flat_params(actor, candidate)
        reward_surr, cost_surr, mean_kl = _evaluate_candidate(
            actor,
            old_actor,
            states_torch,
            actions_torch,
            old_log_prob_torch,
            reward_adv_raw_torch,
            cost_adv_raw_torch,
            old_cost_mean,
        )
        kl_ok = mean_kl <= float(delta) + LINE_SEARCH_TOL
        if mode == "normal":
            accept = kl_ok and (reward_surr >= -LINE_SEARCH_TOL) and (cost_surr <= float(constraint_limit) + LINE_SEARCH_TOL)
        else:
            accept = kl_ok and (
                (cost_surr <= float(constraint_limit) + LINE_SEARCH_TOL)
                or (cost_surr < float(old_cost_mean) - LINE_SEARCH_TOL)
            )
        if accept:
            return {
                "accepted": True,
                "backtracks": backtrack,
                "scale": scale,
                "reward_surrogate": reward_surr,
                "cost_surrogate": cost_surr,
                "mean_kl": mean_kl,
                "mode": mode,
            }
    _set_flat_params(actor, old_params)
    return {
        "accepted": False,
        "backtracks": int(max_backtracks),
        "scale": 0.0,
        "reward_surrogate": 0.0,
        "cost_surrogate": float(old_cost_mean),
        "mean_kl": 0.0,
        "mode": mode,
    }


def _format_metric(value, precision=4):
    numeric = float(value)
    abs_numeric = abs(numeric)
    if (abs_numeric >= 1e4) or ((abs_numeric > 0.0) and (abs_numeric < 1e-3)):
        return "{0:.2e}".format(numeric)
    return "{0:.{1}f}".format(numeric, int(precision))


def _print_iteration_summary(
    iteration_index,
    num_iterations,
    objective_avg,
    cost_avg,
    constraint_violation,
    reward_rate,
    cost_rate,
    search_info,
    cg_count,
):
    print(
        "ACPO iter {0:03d}/{1:03d} | obj={2} | cost={3} | viol={4} | "
        "r_bar={5} | c_bar={6} | mode={7} | accepted={8} | kl={9} | bt={10} | cg={11} | scale={12}".format(
            int(iteration_index) + 1,
            int(num_iterations),
            _format_metric(objective_avg),
            _format_metric(cost_avg),
            _format_metric(constraint_violation),
            _format_metric(reward_rate),
            _format_metric(cost_rate),
            str(search_info["mode"]),
            int(bool(search_info["accepted"])),
            _format_metric(search_info["mean_kl"], precision=6),
            int(search_info["backtracks"]),
            int(cg_count),
            _format_metric(search_info["scale"], precision=6),
        )
    )


def ACPO_main(args, example_name):
    scene_name = _scene_name(example_name)

    seed = int(getattr(args, "seed", 0))
    _set_seed(seed)
    device = resolve_torch_device(getattr(args, "device", None))

    horizon = int(args.T)
    num_iterations = int(args.episode)
    constraint_limit = float(getattr(args, "constraint_limit", _default_constraint_limit(scene_name)))
    gae_lambda_reward = float(args.gae_lambda_reward)
    gae_lambda_cost = float(args.gae_lambda_cost)
    delta = float(args.delta)
    backtrack_coeff = float(args.backtrack_coeff)
    max_backtracks = int(args.max_backtracks)
    cg_iters = int(args.cg_iters)
    damping = float(args.cg_damping)
    recovery_t = float(args.recovery_t)
    vf_lr = float(args.vf_lr)
    vf_epochs = int(args.vf_epochs)
    vf_batch_size = int(args.vf_batch_size)
    init_log_std = float(args.init_log_std)

    env, actor, state_dim, action_dim, cost_scale = _build_scene(
        scene_name,
        seed,
        device,
        horizon,
        init_log_std,
        args,
    )
    observation = env.reset()
    reward_critic = AverageValueCritic(state_dim, device, vf_lr)
    cost_critic = AverageValueCritic(state_dim, device, vf_lr)
    normalizer = RunningNormalizer(state_dim)

    objective_curve = []
    cost_curve = []
    diagnostics = {
        "accepted": [],
        "mode": [],
        "backtracks": [],
        "mean_kl": [],
        "reward_surrogate": [],
        "cost_surrogate": [],
        "constraint_violation": [],
        "cg_iters": [],
        "objective_avg": [],
        "cost_avg": [],
        "reward_rate": [],
        "cost_rate": [],
        "line_search_scale": [],
    }

    for iteration_index in range(num_iterations):
        batch = _rollout_batch(
            env,
            actor,
            normalizer,
            horizon,
            constraint_limit,
            device,
            observation,
            cost_scale,
        )
        observation = batch["last_observation"]
        stats = _prepare_advantages(
            batch,
            reward_critic,
            cost_critic,
            gae_lambda_reward,
            gae_lambda_cost,
            device,
        )

        reward_critic.fit(batch["state_norm"], stats["reward_returns"], vf_epochs, vf_batch_size)
        cost_critic.fit(batch["state_norm"], stats["cost_returns"], vf_epochs, vf_batch_size)

        objective_avg = float(np.mean(batch["objective_cost"]))
        cost_avg = float(np.mean(batch["cost"]))
        objective_curve.append(objective_avg)
        cost_curve.append(cost_avg)

        old_actor = _clone_actor(actor)
        old_params = _get_flat_params(actor).detach().clone()
        states_torch = torch.tensor(batch["state_norm"], dtype=torch.float32, device=device)
        actions_torch = torch.tensor(batch["action"], dtype=torch.float32, device=device)
        reward_adv_pg_torch = torch.tensor(stats["reward_adv_norm"], dtype=torch.float32, device=device)
        cost_adv_pg_torch = torch.tensor(stats["cost_adv_norm"], dtype=torch.float32, device=device)
        reward_adv_raw_torch = torch.tensor(stats["reward_adv_raw"], dtype=torch.float32, device=device)
        cost_adv_raw_torch = torch.tensor(stats["cost_adv_raw"], dtype=torch.float32, device=device)
        old_log_prob_torch = _log_prob(old_actor, states_torch, actions_torch).detach()

        g_torch = _surrogate_grad(actor, states_torch, actions_torch, reward_adv_pg_torch).detach()
        a_torch = _surrogate_grad(actor, states_torch, actions_torch, cost_adv_pg_torch).detach()
        constraint_violation = float(np.mean(batch["cost"]) - constraint_limit)

        hvp_fn = lambda vec: _fisher_vector_product(actor, old_actor, states_torch, vec, damping).detach()
        step_terms, cg_count = _solve_feasible_step(g_torch, a_torch, hvp_fn, delta, cg_iters)

        accepted = False
        search_info = None
        if step_terms is not None:
            feasible_step, _ = _primal_feasible_step(step_terms, constraint_violation, delta, hvp_fn)
            if feasible_step is not None:
                search_info = _line_search(
                    actor,
                    old_actor,
                    old_params,
                    feasible_step,
                    states_torch,
                    actions_torch,
                    old_log_prob_torch,
                    reward_adv_raw_torch,
                    cost_adv_raw_torch,
                    cost_avg,
                    constraint_limit,
                    delta,
                    backtrack_coeff,
                    max_backtracks,
                    mode="normal",
                )
                accepted = bool(search_info["accepted"])

        if not accepted:
            recovery_step, cg_recovery = _recovery_step(g_torch, a_torch, hvp_fn, delta, recovery_t, cg_iters)
            cg_count = max(cg_count, cg_recovery)
            search_info = _line_search(
                actor,
                old_actor,
                old_params,
                recovery_step,
                states_torch,
                actions_torch,
                old_log_prob_torch,
                reward_adv_raw_torch,
                cost_adv_raw_torch,
                cost_avg,
                constraint_limit,
                delta,
                backtrack_coeff,
                max_backtracks,
                mode="recovery",
            )
            accepted = bool(search_info["accepted"])

        if not accepted:
            _set_flat_params(actor, old_params)
            search_info["mode"] = "skipped"

        diagnostics["accepted"].append(int(bool(search_info["accepted"])))
        diagnostics["mode"].append(search_info["mode"])
        diagnostics["backtracks"].append(int(search_info["backtracks"]))
        diagnostics["mean_kl"].append(float(search_info["mean_kl"]))
        diagnostics["reward_surrogate"].append(float(search_info["reward_surrogate"]))
        diagnostics["cost_surrogate"].append(float(search_info["cost_surrogate"]))
        diagnostics["constraint_violation"].append(float(constraint_violation))
        diagnostics["cg_iters"].append(int(cg_count))
        diagnostics["objective_avg"].append(objective_avg)
        diagnostics["cost_avg"].append(cost_avg)
        diagnostics["reward_rate"].append(float(stats["reward_rate"]))
        diagnostics["cost_rate"].append(float(stats["cost_rate"]))
        diagnostics["line_search_scale"].append(float(search_info["scale"]))

        _print_iteration_summary(
            iteration_index,
            num_iterations,
            objective_avg,
            cost_avg,
            constraint_violation,
            float(stats["reward_rate"]),
            float(stats["cost_rate"]),
            search_info,
            cg_count,
        )

    diagnostics_out = {
        "accepted": np.asarray(diagnostics["accepted"], dtype=np.int32),
        "mode": np.asarray(diagnostics["mode"], dtype="U16"),
        "backtracks": np.asarray(diagnostics["backtracks"], dtype=np.int32),
        "mean_kl": np.asarray(diagnostics["mean_kl"], dtype=np.float64),
        "reward_surrogate": np.asarray(diagnostics["reward_surrogate"], dtype=np.float64),
        "cost_surrogate": np.asarray(diagnostics["cost_surrogate"], dtype=np.float64),
        "constraint_violation": np.asarray(diagnostics["constraint_violation"], dtype=np.float64),
        "cg_iters": np.asarray(diagnostics["cg_iters"], dtype=np.int32),
        "objective_avg": np.asarray(diagnostics["objective_avg"], dtype=np.float64),
        "cost_avg": np.asarray(diagnostics["cost_avg"], dtype=np.float64),
        "reward_rate": np.asarray(diagnostics["reward_rate"], dtype=np.float64),
        "cost_rate": np.asarray(diagnostics["cost_rate"], dtype=np.float64),
        "line_search_scale": np.asarray(diagnostics["line_search_scale"], dtype=np.float64),
    }

    return objective_curve, cost_curve, diagnostics_out
