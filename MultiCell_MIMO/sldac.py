import numpy as np
import torch
from datetime import datetime

from .artifact_paths import build_checkpoint_dir
from .buffer import LegacySLDACBuffer
from .checkpoint import save_checkpoint
from .config import validate_config
from .critic import LegacyMultiHeadDifferentialCritic
from .cssca import solve_cssca_update
from .environment import MultiCellMIMOEnv
from .model import SharedLocalGaussianActor
from .seed_utils import resolve_torch_device, set_global_seed
from .tree_critic import TreeMessageDifferentialCritic


def multicell_power_action_to_db_action(action, env):
    env_action = np.asarray(action, dtype=np.float64).reshape(-1).copy()
    action_cells = env_action.reshape(env.cell_count, env.cell_action_dim)
    power = action_cells[:, : env.users_per_cell]
    # dB-action 环境只改变接口坐标；raw power 先对齐 MultiCell legacy floor。
    safe_power = np.maximum(power, env.noise_power)
    action_cells[:, : env.users_per_cell] = 10.0 * np.log10(safe_power / env.noise_power)
    return action_cells.reshape(-1)


def multicell_buffer_action_from_info(action, info, action_interface):
    if "executed_power_action" in info:
        return np.asarray(info["executed_power_action"], dtype=np.float64).reshape(-1).copy()
    return np.asarray(action, dtype=np.float64).reshape(-1).copy()


def _resolve_run_id(config):
    configured = str(config.get("run_id", "")).strip()
    if configured:
        return configured
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")


def _should_log_episode(current_episode, total_episodes, log_interval_episodes):
    current = int(current_episode)
    total = int(total_episodes)
    interval = int(log_interval_episodes)
    return current == 1 or current == total or current % interval == 0


def _print_episode_progress(config, current_episode, total_episodes, objective_average, cost_average):
    print("SLDAC_EPISODE: {0}/{1}".format(int(current_episode), int(total_episodes)), flush=True)
    print("objective_average:", float(objective_average), flush=True)
    print("cost_average:", float(cost_average), flush=True)
    print("critic_backend:", str(config["critic_backend"]), flush=True)
    print("action_interface:", str(config["action_interface"]), flush=True)


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
        return LegacyMultiHeadDifferentialCritic(
            state_dim=env.state_dim,
            action_dim=env.action_dim,
            constraint_dim=env.constraint_dim,
            q_update_time=int(config["q_update_time"]),
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
    gamma_reward,
    gamma_cost,
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
            gamma_reward=gamma_reward,
            gamma_cost=gamma_cost,
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
            gamma=gamma_reward,
            critic_target_mode=config["critic_target_mode"],
        )
    raise ValueError("unsupported critic_backend: {0}".format(config["critic_backend"]))


def _estimate_actor_gradients(actor, critic, env, state_batch, action_batch, constraint_dim, critic_backend):
    device = actor.device
    action_torch = torch.as_tensor(action_batch, dtype=torch.float32, device=device)
    local_state = torch.as_tensor(env.batch_local_actor_observations(state_batch), dtype=torch.float32, device=device)
    critic_state = _critic_state_tensor(env, state_batch, critic_backend, device)
    q_hat = critic.critic_value(critic_state, action_torch, use_target=True)
    q_hat_np = q_hat.detach().cpu().numpy()
    q_hat_np[:, 0] = (q_hat_np[:, 0] - np.mean(q_hat_np[:, 0])) / (np.std(q_hat_np[:, 0]) + 1e-6)
    objective_scale = np.std(q_hat_np[:, 0]) + 1e-6
    for q_idx in range(1, 1 + int(constraint_dim)):
        q_hat_np[:, q_idx] = (q_hat_np[:, q_idx] - np.mean(q_hat_np[:, q_idx])) / objective_scale
    q_hat = torch.as_tensor(q_hat_np, dtype=torch.float32, device=device)

    theta_dim = int(actor.flatten_parameters().numel())
    grad = torch.zeros((1 + int(constraint_dim), theta_dim), dtype=torch.float32, device=device)
    for head_idx in range(1 + int(constraint_dim)):
        actor.clear_policy_grad(set_to_none=True)
        # Match SLDAC_code: use the joint action log-prob directly.
        log_prob = actor.evaluate_action(local_state, action_torch)
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
        action_interface=str(config["action_interface"]),
    )
    observation = env.reset()
    actor = SharedLocalGaussianActor(
        local_state_dim=env.local_actor_state_dim,
        users_per_cell=env.users_per_cell,
        cell_count=env.cell_count,
        hidden_dims=tuple(config["hidden_dims"]),
        device=device,
        power_max=float(config["power_max"]),
        log_std_min=float(config["log_std_min"]),
        log_std_max=float(config["log_std_max"]),
    )
    critic = _build_critic(config, env, device)

    t_horizon = int(config["t_horizon"])
    grad_batch_size = int(config["grad_batch_size"])
    num_new_data = int(config["num_new_data"])
    q_update_time = int(config["q_update_time"])
    critic_update_interval = int(num_new_data // q_update_time)
    update_time_per_episode = int(config["update_time_per_episode"])
    episode_count = int(config["episode"])
    log_interval_episodes = int(config["log_interval_episodes"])
    num_update_time = episode_count * update_time_per_episode
    max_steps = 2 * t_horizon + num_update_time * num_new_data + q_update_time + 1
    buffer = LegacySLDACBuffer(
        t_horizon=t_horizon,
        num_new_data=num_new_data,
        state_dim=env.state_dim,
        action_dim=env.action_dim,
        cost_dim=1 + env.constraint_dim,
        window=int(config["window"]),
    )

    func_value = np.zeros((1 + env.constraint_dim,), dtype=np.float64)
    grad_estimate = np.zeros((1 + env.constraint_dim, int(actor.flatten_parameters().numel())), dtype=np.float64)
    objective_history = []
    cost_history = []
    update_index = 0
    q_update_index = 0
    cssca_info = {}

    for step_idx in range(max_steps):
        state = observation
        local_state = env.local_actor_observations()
        action = actor.sample_action(local_state, use_mean=False)
        env_action = (
            multicell_power_action_to_db_action(action, env)
            if str(config["action_interface"]) == "snr_db"
            else action
        )
        observation, objective_cost, done, info = env.step(env_action)
        _ = done
        costs = _build_cost_vector(objective_cost, info, env.constraint_dim, config["constraint_limit"])
        buffer_action = multicell_buffer_action_from_info(action, info, config["action_interface"])
        buffer.store_experiences(
            state=state,
            action=buffer_action,
            costs=costs,
            next_state=observation,
            aver_objective=objective_cost,
            aver_cost=float(info.get("cost", 0.0)) / float(max(env.constraint_dim, 1)),
        )

        if step_idx <= 2 * t_horizon:
            continue
        if (step_idx - 2 * t_horizon) % critic_update_interval != 0:
            continue

        q_update_index += 1
        alpha = 1.0 / float((update_index + 1) ** float(config["alpha_pow"]))
        beta = 1.0 / float((update_index + 1) ** float(config["beta_pow"]))
        eta = 1.0 / float((update_index + 1) ** float(config["eta_pow"]))
        if q_update_index == q_update_time:
            gamma_reward = 1.0 / float((update_index + 1) ** float(config["gamma_pow_reward"]))
            gamma_cost = 1.0 / float((update_index + 1) ** float(config["gamma_pow_cost"]))
        else:
            gamma_reward = 0.0
            gamma_cost = 0.0

        states_all, actions_all, costs_all, next_states_all, aver_objective, aver_cost = buffer.take_experiences()
        func_value_tilda = costs_all.mean(axis=0)
        func_value = (1.0 - alpha) * func_value + alpha * func_value_tilda

        if update_index % update_time_per_episode == 0 and q_update_index == 1:
            objective_history.append(float(np.mean(aver_objective)))
            cost_history.append(float(np.mean(aver_cost)))
            current_episode = len(objective_history)
            if _should_log_episode(current_episode, episode_count, log_interval_episodes):
                _print_episode_progress(
                    config,
                    current_episode,
                    episode_count,
                    objective_history[-1],
                    cost_history[-1],
                )

        batch_start = max(0, 2 * t_horizon - grad_batch_size)
        state_batch = states_all[batch_start : 2 * t_horizon]
        action_batch = actions_all[batch_start : 2 * t_horizon]
        costs_batch = costs_all[batch_start : 2 * t_horizon]
        next_state_batch = next_states_all[batch_start : 2 * t_horizon]
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
            gamma_reward=gamma_reward,
            gamma_cost=gamma_cost,
        )

        if q_update_index != q_update_time:
            continue
        q_update_index = 0

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
        "action_interface": str(config["action_interface"]),
        "run_id": str(config["run_id"]),
    }
