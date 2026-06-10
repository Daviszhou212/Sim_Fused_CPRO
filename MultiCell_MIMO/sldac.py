import numpy as np
import torch
from datetime import datetime

from .artifact_paths import build_checkpoint_dir
from .buffer import TransitionBuffer
from .checkpoint import save_checkpoint
from .config import validate_config
from .critic import MultiHeadDifferentialCritic
from .cssca import solve_cssca_update
from .environment import MultiCellMIMOEnv
from .model import SharedLocalGaussianActor
from .seed_utils import resolve_torch_device, set_global_seed
from .tree_critic import TreeMessageDifferentialCritic


def _resolve_run_id(config):
    configured = str(config.get("run_id", "")).strip()
    if configured:
        return configured
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _build_cost_vector(objective_cost, info, constraint_dim, constraint_limit):
    costs = np.zeros((1 + int(constraint_dim),), dtype=np.float64)
    costs[0] = float(objective_cost)
    for idx in range(1, int(constraint_dim) + 1):
        costs[idx] = float(info.get("cost_{0}".format(idx), info.get("cost", 0.0))) - float(constraint_limit)
    return costs


def _sample_next_actions(actor, env, next_state_batch):
    actions = []
    for next_state in np.asarray(next_state_batch, dtype=np.float64):
        local_next = env.local_actor_observations_from_state(next_state)
        actions.append(actor.sample_action(local_next, use_mean=False))
    return np.asarray(actions, dtype=np.float64)


def _build_critic(config, env, device):
    if config["critic_backend"] == "centralized":
        return MultiHeadDifferentialCritic(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            constraint_dim=env.constraint_dim,
            hidden_dims=tuple(config["critic_hidden_dims"]),
            device=device,
        )
    if config["critic_backend"] == "tree":
        return TreeMessageDifferentialCritic(
            local_state_dim=env.local_critic_state_dim,
            cell_count=env.cell_count,
            cell_action_dim=env.cell_action_dim,
            constraint_dim=env.constraint_dim,
            message_dim=int(config["tree_message_dim"]),
            hidden_dims=tuple(config["critic_hidden_dims"]),
            device=device,
        )
    raise ValueError("unsupported critic_backend: {0}".format(config["critic_backend"]))


def _critic_state_tensor(env, state_batch, critic_backend, device):
    if critic_backend == "centralized":
        return torch.as_tensor(state_batch, dtype=torch.float32, device=device)
    if critic_backend == "tree":
        local_state = env.batch_local_critic_observations(state_batch)
        return torch.as_tensor(local_state, dtype=torch.float32, device=device)
    raise ValueError("unsupported critic_backend: {0}".format(critic_backend))


def _update_critic(
    critic,
    env,
    config,
    device,
    state_batch,
    action_batch,
    costs_batch,
    next_state_batch,
    next_action_batch,
    func_value,
    eta,
    gamma,
):
    action_torch = torch.as_tensor(action_batch, dtype=torch.float32, device=device)
    next_action_torch = torch.as_tensor(next_action_batch, dtype=torch.float32, device=device)
    costs_torch = torch.as_tensor(costs_batch, dtype=torch.float32, device=device)
    func_value_torch = torch.as_tensor(func_value, dtype=torch.float32, device=device)

    if config["critic_backend"] == "centralized":
        return critic.update(
            state=_critic_state_tensor(env, state_batch, "centralized", device),
            action=action_torch,
            costs=costs_torch,
            next_state=_critic_state_tensor(env, next_state_batch, "centralized", device),
            next_action=next_action_torch,
            func_value=func_value_torch,
            eta=eta,
            gamma=gamma,
            critic_target_mode=config["critic_target_mode"],
        )
    if config["critic_backend"] == "tree":
        return critic.update(
            local_state=_critic_state_tensor(env, state_batch, "tree", device),
            action=action_torch,
            costs=costs_torch,
            next_local_state=_critic_state_tensor(env, next_state_batch, "tree", device),
            next_action=next_action_torch,
            func_value=func_value_torch,
            eta=eta,
            gamma=gamma,
            critic_target_mode=config["critic_target_mode"],
        )
    raise ValueError("unsupported critic_backend: {0}".format(config["critic_backend"]))


def _estimate_actor_gradients(actor, critic, env, state_batch, action_batch, constraint_dim, critic_backend):
    device = actor.device
    action_torch = torch.as_tensor(action_batch, dtype=torch.float32, device=device)
    local_state = torch.as_tensor(env.batch_local_actor_observations(state_batch), dtype=torch.float32, device=device)
    critic_state = _critic_state_tensor(env, state_batch, critic_backend, device)
    q_hat = critic.critic_value(critic_state, action_torch, use_target=True)
    q_hat = q_hat - q_hat.mean(dim=0, keepdim=True)

    theta_dim = int(actor.flatten_parameters().numel())
    grad = torch.zeros((1 + int(constraint_dim), theta_dim), dtype=torch.float32, device=device)
    for head_idx in range(1 + int(constraint_dim)):
        actor.zero_grad(set_to_none=True)
        # 共享 actor 的 joint log-prob 是 cell log-prob 之和；进入 CSSCA 前按 cell 平均。
        log_prob = actor.evaluate_action(local_state, action_torch) / float(max(env.cell_count, 1))
        loss = (q_hat[:, head_idx].detach() * log_prob).mean()
        loss.backward()
        grad[head_idx] = actor.flatten_grad()
    return grad.detach().cpu().numpy()


def run_sldac(config):
    config = validate_config(dict(config))
    config["run_id"] = _resolve_run_id(config)
    seed = set_global_seed(config["seed"])
    device = resolve_torch_device(config.get("device", "cpu"))

    env = MultiCellMIMOEnv(
        seed=seed,
        nt=int(config["nt"]),
        cell_count=int(config["cell_count"]),
        users_per_cell=int(config["users_per_cell"]),
        arrival_upper=float(config["arrival_upper"]),
        queue_max=float(config["queue_max"]),
    )
    observation = env.reset()
    actor = SharedLocalGaussianActor(
        local_state_dim=env.local_actor_state_dim,
        users_per_cell=env.users_per_cell,
        cell_count=env.cell_count,
        hidden_dims=tuple(config["hidden_dims"]),
        device=device,
        power_max=float(config["power_max"]),
    )
    critic = _build_critic(config, env, device)

    t_horizon = int(config["t_horizon"])
    grad_batch_size = int(config["grad_batch_size"])
    num_new_data = int(config["num_new_data"])
    update_time_per_episode = int(config["update_time_per_episode"])
    episode_count = int(config["episode"])
    num_update_time = episode_count * update_time_per_episode
    max_steps = 2 * t_horizon + num_update_time * num_new_data + 1
    buffer = TransitionBuffer(
        capacity=max(2 * t_horizon, int(config["window"])),
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cost_dim=1 + env.constraint_dim,
    )

    func_value = np.zeros((1 + env.constraint_dim,), dtype=np.float64)
    grad_estimate = np.zeros((1 + env.constraint_dim, int(actor.flatten_parameters().numel())), dtype=np.float64)
    objective_history = []
    cost_history = []
    update_index = 0

    for step_idx in range(max_steps):
        state = observation
        local_state = env.local_actor_observations()
        action = actor.sample_action(local_state, use_mean=False)
        observation, objective_cost, done, info = env.step(action)
        _ = done
        costs = _build_cost_vector(objective_cost, info, env.constraint_dim, config["constraint_limit"])
        buffer.store(state, action, costs, observation)

        if len(buffer) < 2 * t_horizon:
            continue
        if (step_idx - 2 * t_horizon) % num_new_data != 0:
            continue

        alpha = 1.0 / float((update_index + 1) ** float(config["alpha_pow"]))
        beta = 1.0 / float((update_index + 1) ** float(config["beta_pow"]))
        eta = 1.0 / float((update_index + 1) ** float(config["eta_pow"]))
        gamma = 1.0 / float((update_index + 1) ** float(config["gamma_pow_reward"]))

        states_all, _actions_all, costs_all, _next_all = buffer.arrays()
        func_value_tilda = costs_all.mean(axis=0)
        func_value = (1.0 - alpha) * func_value + alpha * func_value_tilda

        if update_index % update_time_per_episode == 0:
            objective_history.append(float(np.mean(costs_all[:, 0])))
            cost_history.append(float(np.mean(costs_all[:, 1:] + float(config["constraint_limit"]))))

        state_batch, action_batch, costs_batch, next_state_batch = buffer.latest(grad_batch_size)
        next_action_batch = _sample_next_actions(actor, env, next_state_batch)
        _update_critic(
            critic=critic,
            env=env,
            config=config,
            device=device,
            state_batch=state_batch,
            action_batch=action_batch,
            costs_batch=costs_batch,
            next_state_batch=next_state_batch,
            next_action_batch=next_action_batch,
            func_value=func_value,
            eta=eta,
            gamma=gamma,
        )

        grad_tilda = _estimate_actor_gradients(
            actor,
            critic,
            env,
            state_batch,
            action_batch,
            env.constraint_dim,
            config["critic_backend"],
        )
        grad_estimate = (1.0 - alpha) * grad_estimate + alpha * grad_tilda

        theta = actor.flatten_parameters().detach().cpu().numpy()
        theta_bar, cssca_info = solve_cssca_update(
            func_value,
            grad_estimate,
            theta,
            tau_objective=float(config["tau_objective"]),
            tau_constraint=float(config["tau_constraint"]),
            solver=str(config["cssca_solver"]),
        )
        theta_next = (1.0 - beta) * theta + beta * theta_bar
        actor.restore_parameters(torch.as_tensor(theta_next, dtype=torch.float32, device=device))

        update_index += 1
        if len(objective_history) >= episode_count:
            if int(config.get("save_final_checkpoint", 0)):
                checkpoint_dir = build_checkpoint_dir(
                    config["checkpoint_root"],
                    "SLDAC",
                    config.get("run_tag", "multicell_sldac"),
                    seed,
                    config.get("run_id", ""),
                )
                state_dict = {}
                state_dict.update({"actor." + key: value for key, value in actor.checkpoint_state().items()})
                state_dict.update({"critic." + key: value for key, value in critic.checkpoint_state().items()})
                save_checkpoint(
                    checkpoint_root=checkpoint_dir,
                    config=config,
                    state_dict=state_dict,
                    stats={
                        "objective_history": objective_history,
                        "cost_history": cost_history,
                        "cssca_last": cssca_info,
                    },
                    episode_index=len(objective_history),
                    reason="final",
                )
            break

    return {
        "objective_history": np.asarray(objective_history, dtype=np.float64),
        "cost_history": np.asarray(cost_history, dtype=np.float64),
        "func_value": np.asarray(func_value, dtype=np.float64),
        "critic_backend": str(config["critic_backend"]),
        "run_id": str(config["run_id"]),
    }
