from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .cssca_solvers import CsscaProblem, solve_cssca
from .mimo1_buffer import DataStorage
from .mimo1_critic import Critic
from .mimo1_environment import EnvironmentMIMO
from .mimo1_model import GaussianPolicyMIMO


@dataclass
class SpanSldacConfig:
    seed: int = 0
    T: int = 500
    grad_T: int = 500
    num_new_data: int = 100
    episode: int = 60
    update_time_per_episode: int = 10
    num_update_time: int = 600
    q_update_time: int = 1
    window: int = 10000
    print_interval: int = 1
    alpha_pow: float = 0.6
    beta_pow: float = 0.7
    eta_pow: float = 0.01
    gamma_pow_reward: float = 0.3
    gamma_pow_cost: float = 0.3
    tau_reward: float = 1.0
    tau_cost: float = 1.0
    device: str = "cpu"

    @property
    def max_steps(self) -> int:
        return int(2 * self.T + self.num_update_time * self.num_new_data)


def _normalise_q_hat(q_hat):
    q_np = q_hat.detach().cpu().numpy()
    q_np[:, 0] = (q_np[:, 0] - np.mean(q_np[:, 0])) / (np.std(q_np[:, 0]) + 1e-6)
    for idx in range(1, q_np.shape[1]):
        q_np[:, idx] = (q_np[:, idx] - np.mean(q_np[:, idx])) / (np.std(q_np[:, 0]) + 1e-6)
    return q_np


def _log(verbose: bool, message: str):
    if verbose:
        print(message, flush=True)


def run_span_sldac(solver_name: str, config: SpanSldacConfig, verbose: bool = True):
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    device = config.device.lower()
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"

    env = EnvironmentMIMO(seed=config.seed, nt=8, ue_num=4)
    state_dim = env.state_dim
    action_dim = env.action_dim
    constraint_dim = 4
    constraint_limit = np.array([1.2, 1.2, 1.2, 1.2], dtype=np.float64)
    actor = GaussianPolicyMIMO(state_dim, action_dim, device, config.grad_T)
    buffer = DataStorage(config.T, config.num_new_data, state_dim, action_dim, constraint_dim, config.window)
    critic = Critic(state_dim, action_dim, constraint_dim, config.q_update_time, device)

    theta_torch = actor.flat_parameters().detach().clone()
    theta_dim = int(theta_torch.numel())
    func_value = np.zeros(constraint_dim + 1, dtype=np.float64)
    grad = np.zeros((constraint_dim + 1, theta_dim), dtype=np.float64)

    reward_curve = []
    cost_curve = []
    timing_rows = []
    observation = env.reset()
    update_index = 0
    episode_index = 0
    q_update_index = 0
    print_interval = max(1, int(config.print_interval))

    _log(
        verbose,
        (
            f"[Spantest][{solver_name}] start "
            f"max_steps={config.max_steps} T={config.T} grad_T={config.grad_T} "
            f"num_new_data={config.num_new_data} episodes={config.episode} "
            f"theta_dim={theta_dim} print_interval={print_interval}"
        ),
    )

    for step_idx in range(config.max_steps):
        state = observation
        action = actor.sample_action(state)
        observation, objective_cost, _, info = env.step(action)
        costs = np.zeros(constraint_dim + 1)
        costs[0] = objective_cost
        for idx in range(1, constraint_dim + 1):
            costs[idx] = info.get("cost_" + str(idx), info.get("cost", 0.0)) - constraint_limit[idx - 1]
        avg_constraint_cost = info.get("cost", 0.0) / constraint_dim
        buffer.store(state, action, costs, observation, objective_cost, avg_constraint_cost)

        if step_idx > 2 * config.T and ((step_idx - 2 * config.T) % (config.num_new_data / config.q_update_time) == 0):
            q_update_index += 1
            alpha = 1 / ((update_index + 1) ** config.alpha_pow)
            beta = 1 / ((update_index + 1) ** config.beta_pow)
            eta = 1 / ((update_index + 1) ** config.eta_pow)
            if q_update_index == config.q_update_time:
                gamma_reward = 1 / ((update_index + 1) ** config.gamma_pow_reward)
                gamma_cost = 1 / ((update_index + 1) ** config.gamma_pow_cost)
            else:
                gamma_reward = 0.0
                gamma_cost = 0.0

            state_buffer, action_buffer, costs_buffer, next_state_buffer, reward_window, cost_window = buffer.take()
            func_value = (1 - alpha) * func_value + alpha * np.mean(costs_buffer, axis=0)
            should_print_episode = False
            if (update_index % config.update_time_per_episode == 0) and (q_update_index == 1):
                reward_curve.append(float(np.mean(reward_window)))
                cost_curve.append(float(np.mean(cost_window)))
                episode_index += 1
                should_print_episode = episode_index % print_interval == 0

            state_batch = state_buffer[(2 * config.T - config.grad_T) : 2 * config.T]
            action_batch = action_buffer[(2 * config.T - config.grad_T) : 2 * config.T]
            costs_batch = costs_buffer[(2 * config.T - config.grad_T) : 2 * config.T]
            next_state_batch = next_state_buffer[(2 * config.T - config.grad_T) : 2 * config.T]
            next_action_batch = np.zeros((config.grad_T, action_dim))
            for idx in range(config.grad_T):
                next_action_batch[idx] = actor.sample_action(next_state_batch[idx])

            critic.update(
                func_value,
                state_batch,
                action_batch,
                costs_batch,
                next_state_batch,
                next_action_batch,
                eta,
                gamma_reward,
                gamma_cost,
            )

            if q_update_index == config.q_update_time:
                update_index += 1
                q_update_index = 0
                state_t = torch.tensor(state_batch, dtype=torch.float, device=device)
                action_t = torch.tensor(action_batch, dtype=torch.float, device=device)
                q_hat_np = _normalise_q_hat(critic.value(state_t, action_t))
                q_hat_t = torch.tensor(q_hat_np, dtype=torch.float, device=device)
                grad_tilde = torch.zeros((1 + constraint_dim, theta_dim), dtype=torch.float, device=device)
                for head_idx in range(1 + constraint_dim):
                    actor.zero_grad()
                    log_prob = actor.evaluate_action(state_t, action_t)
                    loss = (q_hat_t[:, head_idx] * log_prob).mean()
                    loss.backward()
                    grad_tilde[head_idx] = actor.flat_gradient()
                grad = (1 - alpha) * grad + alpha * grad_tilde.detach().cpu().numpy()

                problem = CsscaProblem(
                    values=func_value.copy(),
                    gradients=grad.copy(),
                    theta=theta_torch.detach().cpu().numpy(),
                    tau=np.array([config.tau_reward] + [config.tau_cost] * constraint_dim, dtype=np.float64),
                )
                result = solve_cssca(problem, solver_name)
                theta_bar_t = torch.tensor(result.theta_bar, dtype=torch.float, device=device)
                theta_torch = (1 - beta) * theta_torch + beta * theta_bar_t
                actor.set_flat_parameters(theta_torch)
                timing_rows.append(
                    {
                        "solver": solver_name,
                        "update": update_index,
                        "branch": result.branch,
                        "status": result.status,
                        "theta_dim": theta_dim,
                        "decision_dim": result.decision_dim,
                        "gradient_rank": result.gradient_rank,
                        "solve_time_sec": result.solve_time_sec,
                        "feasible_x": result.feasible_x,
                        "objective_value": result.objective_value,
                        "active_rows": " ".join(str(item) for item in result.active_rows),
                    }
                )
                if should_print_episode:
                    _log(
                        verbose,
                        (
                            f"[Spantest][{solver_name}] episode={episode_index}/{config.episode} "
                            f"update={update_index}/{config.num_update_time} "
                            f"branch={result.branch} status={result.status} "
                            f"decision_dim={result.decision_dim} gradient_rank={result.gradient_rank} "
                            f"solve_time_sec={result.solve_time_sec:.6f} "
                            f"objective_cost={reward_curve[-1]:.6f} "
                            f"constraint_cost={cost_curve[-1]:.6f}"
                        ),
                    )

    _log(
        verbose,
        f"[Spantest][{solver_name}] finished updates={len(timing_rows)} episodes_logged={len(reward_curve)}",
    )

    return {
        "solver": solver_name,
        "reward_curve": np.asarray(reward_curve, dtype=np.float64),
        "cost_curve": np.asarray(cost_curve, dtype=np.float64),
        "timing_rows": timing_rows,
        "theta_dim": theta_dim,
    }
