import json
import os
from dataclasses import dataclass

import numpy as np
import torch
from scipy.io import savemat

from .artifact_paths import ensure_dir, make_compare_run_id, make_run_paths
from .bayesian_critic import BayesianCritic, normalize_q_hat_like_sldac_code, risk_correct_q_values
from .buffer import DataStorage
from .config import make_run_config
from .environment import Environment_MIMO
from .lagrangian_cssca import update_policy
from .model import GaussianPolicy_MIMO, actor_to_vector, flatten_actor_grad, vector_to_actor


@dataclass
class SldacRunResult:
    reward_average_save: list
    cost_average_save: list
    diagnostics: dict


@dataclass
class CompareResult:
    output_dir: str
    summary: dict


def _build_costs(reward, info, constraint_dim, constr_lim):
    costs = np.zeros(constraint_dim + 1, dtype=np.float64)
    costs[0] = reward
    for idx in range(1, constraint_dim + 1):
        costs[idx] = info.get("cost_" + str(idx), info.get("cost", 0.0)) - constr_lim[idx - 1]
    return costs


def _prepare_next_actions(actor, next_state_buffer, start, grad_t, action_dim):
    next_action_batch = np.zeros((grad_t, action_dim), dtype=np.float64)
    for idx in range(grad_t):
        next_action_batch[idx, :] = actor.sample_action(next_state_buffer[start + idx, :])
    return next_action_batch


def _estimate_actor_gradient_rows(actor, q_hat_torch, state_batch_torch, action_batch_torch, constraint_dim):
    grad_tilda = np.zeros((int(constraint_dim) + 1, actor_to_vector(actor).size), dtype=np.float64)
    for head_idx in range(int(constraint_dim) + 1):
        # 旧 SLDAC 只调用 actor.zero_grad()，log_std 不是 Parameter，梯度会在本轮各 head 间累积。
        actor.zero_grad()
        log_prob = actor.evaluate_action(state_batch_torch, action_batch_torch)
        actor_loss = (q_hat_torch[:, head_idx] * log_prob).mean()
        actor_loss.backward()
        grad_tilda[head_idx] = flatten_actor_grad(actor)
    return grad_tilda


def SLDAC_main(config, example_name="MIMO", mode="bayesian"):
    if "MIMO" not in str(example_name):
        raise ValueError("Bayesian_SLDAC_MIMO only supports MIMO")
    seed = int(config.seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = str(config.device)

    t_horizon = int(config.T)
    grad_t = int(config.grad_T)
    num_new_data = int(config.num_new_data)
    update_time_per_episode = int(config.update_time_per_episode)
    max_steps = int(config.MAX_STEPS)
    q_update_time = int(config.Q_update_time)
    window = int(config.window)

    nt, ue_num = 8, 4
    state_dim = 2 * ue_num * nt + ue_num
    action_dim = ue_num + 1
    constraint_dim = ue_num
    constr_lim = [1.2, 1.2, 1.2, 1.2]
    env = Environment_MIMO(seed=seed, Nt=nt, UE_num=ue_num)
    actor = GaussianPolicy_MIMO(state_dim, action_dim, device, grad_t)
    buffer = DataStorage(t_horizon, num_new_data, state_dim, action_dim, constraint_dim, window, 1)
    ensemble_size = 1 if str(mode) == "legacy" else int(config.ensemble_size)
    beta_uncertainty = 0.0 if str(mode) == "legacy" else float(config.beta_uncertainty)
    critic = BayesianCritic(
        example_name,
        grad_t,
        state_dim,
        action_dim,
        constraint_dim,
        q_update_time,
        device,
        ensemble_size=ensemble_size,
        bootstrap_mask_prob=1.0 if ensemble_size == 1 else float(config.bootstrap_mask_prob),
        critic_seed=int(config.critic_seed),
        ensemble_init_mode=str(config.ensemble_init_mode),
        critic_lr_base=float(config.critic_lr_base),
    )

    theta = actor_to_vector(actor)
    func_value = np.zeros(constraint_dim + 1, dtype=np.float64)
    grad = np.zeros((constraint_dim + 1, theta.size), dtype=np.float64)
    reward_average_save = []
    cost_average_save = []
    diagnostics = {
        "objective_avg": [],
        "average_cost": [],
        "average_cost_violation": [],
        "constraint_violation": [],
        "worst_user_constraint_residual": [],
        "q_std_objective": [],
        "q_std_constraints": [],
        "q_used_batch_std": [],
        "q_saturation_fraction": [],
        "cssca_status": [],
        "cssca_step_norm": [],
        "func_value": [],
        "grad_norm_objective": [],
        "grad_norm_constraints": [],
        "ensemble_size": int(ensemble_size),
        "beta_uncertainty": float(beta_uncertainty),
        "mode": str(mode),
    }

    observation = env.reset()
    update_index = 0
    print_index = 0
    q_update_index = 0
    for step_idx in range(max_steps):
        state = observation
        action = actor.sample_action(state)
        observation, reward, _done, info = env.step(action)
        next_state = observation
        costs = _build_costs(reward, info, constraint_dim, constr_lim)
        aver_reward = reward
        aver_cost = info.get("cost", 0.0) / constraint_dim
        buffer.store_experiences(state, action, costs, next_state, aver_reward, aver_cost)

        cadence = num_new_data / q_update_time
        if step_idx > 2 * t_horizon and ((step_idx - 2 * t_horizon) % cadence == 0):
            q_update_index += 1
            alpha = 1.0 / ((update_index + 1) ** float(config.alpha_pow))
            beta = 1.0 / ((update_index + 1) ** float(config.beta_pow))
            eta = 1.0 / ((update_index + 1) ** float(config.eta_pow))
            if q_update_index == q_update_time:
                gamma_reward = 1.0 / ((update_index + 1) ** float(config.gamma_pow_reward))
                gamma_cost = 1.0 / ((update_index + 1) ** float(config.gamma_pow_cost))
            else:
                gamma_reward = 0.0
                gamma_cost = 0.0

            state_buffer, action_buffer, costs_buffer, next_state_buffer, aver_reward_batch, aver_cost_batch = buffer.take_experiences()
            func_value_tilda = np.mean(costs_buffer, axis=0)
            func_value = (1.0 - alpha) * func_value + alpha * func_value_tilda

            if (update_index % update_time_per_episode == 0) and (q_update_index == 1):
                reward_average_save.append(float(np.mean(aver_reward_batch)))
                cost_average_save.append(float(np.mean(aver_cost_batch)))
                average_cost = float(np.mean(aver_cost_batch))
                worst_user_residual = float(np.max(np.mean(costs_buffer[:, 1:], axis=0)))
                diagnostics["objective_avg"].append(float(np.mean(aver_reward_batch)))
                diagnostics["average_cost"].append(average_cost)
                diagnostics["average_cost_violation"].append(average_cost - 1.2)
                diagnostics["constraint_violation"].append(worst_user_residual)
                diagnostics["worst_user_constraint_residual"].append(worst_user_residual)
                print_index += 1

            start = 2 * t_horizon - grad_t
            state_batch = state_buffer[start : 2 * t_horizon]
            action_batch = action_buffer[start : 2 * t_horizon]
            costs_batch = costs_buffer[start : 2 * t_horizon]
            next_state_batch = next_state_buffer[start : 2 * t_horizon]
            next_action_batch = _prepare_next_actions(actor, next_state_buffer, start, grad_t, action_dim)
            critic.critic_update(func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost)

            if q_update_index == q_update_time:
                update_index += 1
                q_update_index = 0
                state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=device)
                action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)
                q_mean_torch, q_std_torch = critic.critic_value_stats(state_batch_torch, action_batch_torch)
                q_mean = q_mean_torch.detach().cpu().numpy()
                q_std = q_std_torch.detach().cpu().numpy()
                q_used = risk_correct_q_values(q_mean, q_std, beta_uncertainty=beta_uncertainty)
                diagnostics["q_used_batch_std"].append([float(x) for x in np.std(q_used, axis=0)])
                diagnostics["q_saturation_fraction"].append([float(x) for x in np.mean(np.abs(q_used) >= 199.0, axis=0)])
                q_hat = normalize_q_hat_like_sldac_code(q_used)
                diagnostics["q_std_objective"].append(float(np.mean(q_std[:, 0])))
                diagnostics["q_std_constraints"].append([float(x) for x in np.mean(q_std[:, 1:], axis=0)])

                q_hat_torch = torch.tensor(q_hat, dtype=torch.float, device=device)
                grad_tilda = _estimate_actor_gradient_rows(actor, q_hat_torch, state_batch_torch, action_batch_torch, constraint_dim)
                grad = (1.0 - alpha) * grad + alpha * grad_tilda
                diagnostics["func_value"].append([float(x) for x in func_value])
                diagnostics["grad_norm_objective"].append(float(np.linalg.norm(grad[0])))
                diagnostics["grad_norm_constraints"].append([float(x) for x in np.linalg.norm(grad[1:], axis=1)])

                cssca_result = update_policy(
                    func_value,
                    grad,
                    theta,
                    tau_reward=float(config.tau_reward),
                    tau_cost=float(config.tau_cost),
                    return_info=True,
                )
                diagnostics["cssca_status"].append(cssca_result.status)
                diagnostics["cssca_step_norm"].append(float(cssca_result.step_norm))
                theta = (1.0 - beta) * theta + beta * cssca_result.theta_bar
                vector_to_actor(actor, theta)

    return SldacRunResult(reward_average_save, cost_average_save, diagnostics)


def _diagnostics_to_jsonable(diagnostics):
    out = {}
    for key, value in diagnostics.items():
        if isinstance(value, np.ndarray):
            out[key] = value.tolist()
        else:
            out[key] = value
    return out


def _summarize(run_result):
    average_cost_final = None if not run_result.cost_average_save else float(run_result.cost_average_save[-1])
    average_cost_violation_final = None if average_cost_final is None else float(average_cost_final - 1.2)
    worst_residual = run_result.diagnostics.get("worst_user_constraint_residual", [])
    worst_residual_final = None if not worst_residual else float(worst_residual[-1])
    return {
        "objective_avg_final": None if not run_result.reward_average_save else float(run_result.reward_average_save[-1]),
        "average_cost_final": average_cost_final,
        "average_cost_violation_final": average_cost_violation_final,
        "worst_user_constraint_residual_final": worst_residual_final,
        "constraint_violation_final": worst_residual_final,
        "episodes_recorded": len(run_result.reward_average_save),
        "ensemble_size": int(run_result.diagnostics["ensemble_size"]),
        "beta_uncertainty": float(run_result.diagnostics["beta_uncertainty"]),
    }


def run_compare(config=None, output_root=None):
    cfg = config or make_run_config("b100_q1")
    if output_root is None:
        output_dir = ensure_dir(make_run_paths(make_compare_run_id(cfg.run_tag, cfg.episode)).output_dir)
    else:
        output_dir = ensure_dir(os.path.abspath(output_root))
    legacy = SLDAC_main(cfg, cfg.example_name, mode="legacy")
    bayesian = SLDAC_main(cfg, cfg.example_name, mode="bayesian")

    savemat(os.path.join(output_dir, "legacy_reward_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(legacy.reward_average_save, dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "legacy_cost_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(legacy.cost_average_save, dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "legacy_worst_constraint_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(legacy.diagnostics.get("worst_user_constraint_residual", []), dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "bayesian_reward_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(bayesian.reward_average_save, dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "bayesian_cost_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(bayesian.cost_average_save, dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "bayesian_worst_constraint_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(bayesian.diagnostics.get("worst_user_constraint_residual", []), dtype=np.float64).reshape(1, -1)})

    summary = {
        "run_tag": cfg.run_tag,
        "episode": int(cfg.episode),
        "legacy": _summarize(legacy),
        "bayesian": _summarize(bayesian),
        "diagnostics": {
            "legacy": _diagnostics_to_jsonable(legacy.diagnostics),
            "bayesian": _diagnostics_to_jsonable(bayesian.diagnostics),
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return CompareResult(output_dir=output_dir, summary=summary)


def run_bayesian(config=None, output_root=None):
    cfg = config or make_run_config("b100_q1")
    if output_root is None:
        output_dir = ensure_dir(make_run_paths("bayesian_{0}".format(make_compare_run_id(cfg.run_tag, cfg.episode))).output_dir)
    else:
        output_dir = ensure_dir(os.path.abspath(output_root))
    bayesian = SLDAC_main(cfg, cfg.example_name, mode="bayesian")

    savemat(os.path.join(output_dir, "bayesian_reward_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(bayesian.reward_average_save, dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "bayesian_cost_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(bayesian.cost_average_save, dtype=np.float64).reshape(1, -1)})
    savemat(os.path.join(output_dir, "bayesian_worst_constraint_{0}.mat".format(cfg.run_tag)), {"array": np.asarray(bayesian.diagnostics.get("worst_user_constraint_residual", []), dtype=np.float64).reshape(1, -1)})

    summary = {
        "run_tag": cfg.run_tag,
        "episode": int(cfg.episode),
        "bayesian": _summarize(bayesian),
        "diagnostics": {
            "bayesian": _diagnostics_to_jsonable(bayesian.diagnostics),
        },
    }
    with open(os.path.join(output_dir, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return CompareResult(output_dir=output_dir, summary=summary)
