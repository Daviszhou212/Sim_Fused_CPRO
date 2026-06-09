from environment import Environment_MIMO
from environment import Environment_CLQR
from critic_opt import Critic
from qprop_critic import QPropCritic
from utils import update_policy
from model import GaussianPolicy_MIMO
from model import GaussianPolicy_CLQR
from buffer import DataStorage
import os
import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
PATHWISE_ALGORITHM_NAME = "SLDAC_Pathwise"
POLICY_GRADIENT_MODES = ("stochastic_pathwise", "deterministic_dpg", "qprop_conservative")
BEHAVIOR_POLICY_MODES = ("gaussian_sample", "mean_action")
CHECKPOINT_CONFIG_FIELDS = (
	"T",
	"grad_T",
	"num_new_data",
	"window",
	"episode",
	"update_time_per_episode",
	"num_update_time",
	"Q_update_time",
	"MAX_STEPS",
	"alpha_pow",
	"beta_pow",
	"eta_pow",
	"gamma_pow_reward",
	"gamma_pow_cost",
	"tau_reward",
	"tau_cost",
	"policy_gradient_mode",
	"behavior_policy_mode",
	"normalize_actor_gradient",
	"update_log_std",
	"print_actor_grad_norm",
	"save_diagnostics",
	"use_qprop_dedicated_critic",
	"qprop_critic_update_steps",
	"qprop_replay_batch_size",
	"qprop_target_action_mode",
	"qprop_critic_lr_scale",
	"qprop_target_tau_reward",
	"qprop_target_tau_cost",
)


def _format_seed_dir(seed):
	return "seed_{0}".format(int(seed))


def _arg_to_bool(value):
	if isinstance(value, str):
		return value.strip().lower() not in {"", "0", "false", "no", "off"}
	return bool(value)


def _arg_to_optional_int(value):
	if value is None:
		return None
	if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
		return None
	return int(value)


def _arg_to_optional_float(value):
	if value is None:
		return None
	if isinstance(value, str) and value.strip().lower() in {"", "none", "null"}:
		return None
	return float(value)


def _normalize_config_value(value):
	if isinstance(value, np.generic):
		return value.item()
	if isinstance(value, np.ndarray):
		return value.tolist()
	return value


def _cpu_state_dict(module):
	state_dict_cpu = {}
	for key, value in module.state_dict().items():
		state_dict_cpu[key] = value.detach().cpu().clone()
	return state_dict_cpu


def _get_run_tag(args):
	run_tag = getattr(args, "run_tag", None)
	if run_tag:
		return str(run_tag)
	return "T{0}_G{1}_N{2}_Q{3}".format(int(args.T), int(args.grad_T), int(args.num_new_data), int(args.Q_update_time))


def _get_checkpoint_dir(args, example_name, run_tag, seed):
	root = getattr(args, "checkpoint_root", os.path.join("checkpoints", PATHWISE_ALGORITHM_NAME))
	if not root:
		root = os.path.join("checkpoints", PATHWISE_ALGORITHM_NAME)
	root = str(root)
	if not os.path.isabs(root):
		root = os.path.join(os.getcwd(), root)
	return os.path.join(root, str(example_name), run_tag, _format_seed_dir(seed))


def _collect_config_snapshot(args):
	config = {}
	for field_name in CHECKPOINT_CONFIG_FIELDS:
		config[field_name] = _normalize_config_value(getattr(args, field_name, None))
	return config


def _validate_choice(name, value, valid_values):
	text = str(value).strip()
	if text not in valid_values:
		raise ValueError("{0} must be one of {1}, got {2!r}".format(name, valid_values, value))
	return text


def _resolve_algorithm_modes(args):
	policy_gradient_mode = _validate_choice(
		"policy_gradient_mode",
		getattr(args, "policy_gradient_mode", POLICY_GRADIENT_MODES[0]),
		POLICY_GRADIENT_MODES,
	)
	behavior_policy_mode = _validate_choice(
		"behavior_policy_mode",
		getattr(args, "behavior_policy_mode", BEHAVIOR_POLICY_MODES[0]),
		BEHAVIOR_POLICY_MODES,
	)
	# 训练入口负责显式设置该开关；这里仅保留缺省兜底，避免直接调用时出错。
	normalize_actor_gradient = _arg_to_bool(getattr(args, "normalize_actor_gradient", False))
	update_log_std = _arg_to_bool(getattr(args, "update_log_std", True))
	print_actor_grad_norm = _arg_to_bool(getattr(args, "print_actor_grad_norm", False))
	return policy_gradient_mode, behavior_policy_mode, normalize_actor_gradient, update_log_std, print_actor_grad_norm


def _build_scene(example_name, seed, device, grad_t):
	if "MIMO" in example_name:
		Nt, UE_num = 8, 4
		state_dim = 2 * UE_num * Nt + UE_num
		action_dim = UE_num + 1
		env = Environment_MIMO(seed=seed, Nt=Nt, UE_num=UE_num)
		constraint_dim = UE_num
		constr_lim = [1.2, 1.2, 1.2, 1.2]
		actor = GaussianPolicy_MIMO(state_dim, action_dim, device, grad_t)
	else:
		state_dim, action_dim = 15, 4
		env = Environment_CLQR(seed=seed, state_dim=state_dim, action_dim=action_dim)
		constraint_dim = 1
		constr_lim = 380 * np.ones(constraint_dim)
		actor = GaussianPolicy_CLQR(state_dim, action_dim, device, grad_t)
	return env, actor, state_dim, action_dim, constraint_dim, constr_lim


def _flatten_actor_parameters(actor):
	theta_dim = 0
	for para in actor.net.parameters():
		theta_dim += para.numel()
	real_theta_dim = theta_dim + actor.action_dim
	paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=actor.device)
	ind = 0
	for para in actor.net.parameters():
		tmp = para.numel()
		paras_torch[ind: ind + tmp] = para.detach().view(-1)
		ind = ind + tmp
	paras_torch[ind:] = actor.log_std.detach()
	return paras_torch, real_theta_dim


def _load_actor_parameters(actor, paras_torch, update_log_std):
	ind = 0
	for para in actor.net.parameters():
		tmp = para.numel()
		para.data = paras_torch[ind: ind + tmp].view(para.shape).detach().clone()
		ind = ind + tmp
	if update_log_std:
		actor.log_std = paras_torch[ind:].detach().clone()


def _zero_actor_gradients(actor):
	actor.zero_grad()
	if getattr(actor, "log_std", None) is not None and getattr(actor.log_std, "grad", None) is not None:
		actor.log_std.grad.zero_()


def _extract_actor_gradient(actor, real_theta_dim, normalize_actor_gradient, update_log_std):
	grad_tmp = torch.zeros(real_theta_dim, dtype=torch.float, device=actor.device)
	ind = 0
	for para in actor.net.parameters():
		tmp = para.numel()
		if para.grad is not None:
			grad_tmp[ind: ind + tmp] = para.grad.view(-1)
		ind = ind + tmp
	if update_log_std and getattr(actor.log_std, "grad", None) is not None:
		grad_tmp[ind:] = actor.log_std.grad.view(-1)
	if normalize_actor_gradient:
		grad_scale = torch.linalg.norm(grad_tmp)
		if torch.isfinite(grad_scale) and float(grad_scale.item()) > 1e-8:
			grad_tmp = grad_tmp / grad_scale
	return grad_tmp


def _build_behavior_action(actor, state, behavior_policy_mode):
	use_mean = (behavior_policy_mode == "mean_action")
	return actor.sample_action(state, use_mean=use_mean)


def _build_behavior_action_batch(actor, state_batch, action_dim, behavior_policy_mode):
	next_action_batch = np.zeros((state_batch.shape[0], action_dim))
	for idx in range(state_batch.shape[0]):
		next_action_batch[idx, :] = _build_behavior_action(actor, state_batch[idx, :], behavior_policy_mode)
	return next_action_batch


def _critic_head_value(critic, state_batch_torch, action_batch_torch, head_idx):
	net = getattr(critic, "target_net{0}".format(int(head_idx)))
	return torch.squeeze(net.forward(state_batch_torch, action_batch_torch))


def _control_critic_head_value(critic, state_batch_torch, action_batch_torch, head_idx):
	if hasattr(critic, "head_value"):
		return critic.head_value(head_idx, state_batch_torch, action_batch_torch, use_target=True)
	return _critic_head_value(critic, state_batch_torch, action_batch_torch, head_idx).view(-1)


def _empty_pathwise_diagnostics(head_count, constraint_dim):
	return {
		"q_mean": np.zeros(head_count, dtype=np.float64),
		"q_std": np.zeros(head_count, dtype=np.float64),
		"grad_a_norm": np.zeros(head_count, dtype=np.float64),
		"actor_grad_norm": np.zeros(head_count, dtype=np.float64),
		"constraint_to_objective_grad_norm_ratio": np.zeros(constraint_dim, dtype=np.float64),
		"score_grad_norm": np.zeros(head_count, dtype=np.float64),
		"pathwise_grad_norm": np.zeros(head_count, dtype=np.float64),
		"combined_grad_norm": np.zeros(head_count, dtype=np.float64),
		"qprop_eta": np.zeros(head_count, dtype=np.float64),
		"qprop_covariance": np.zeros(head_count, dtype=np.float64),
		"score_signal_mean": np.zeros(head_count, dtype=np.float64),
		"score_signal_std": np.zeros(head_count, dtype=np.float64),
		"control_signal_mean": np.zeros(head_count, dtype=np.float64),
		"control_signal_std": np.zeros(head_count, dtype=np.float64),
		"qprop_critic_loss": np.zeros(head_count, dtype=np.float64),
		"qprop_critic_td_error_mean": np.zeros(head_count, dtype=np.float64),
		"qprop_critic_target_mean": np.zeros(head_count, dtype=np.float64),
		"qprop_critic_pred_mean": np.zeros(head_count, dtype=np.float64),
		"qprop_replay_batch_size": 0.0,
		"qprop_control_source_code": 0.0,
		"qprop_target_action_mode_code": 0.0,
		"qprop_pathwise_grad_ratio": np.zeros(head_count, dtype=np.float64),
		"qprop_score_grad_ratio": np.zeros(head_count, dtype=np.float64),
	}


def _pack_pathwise_diagnostics(diagnostics_history, constraint_dim):
	head_count = 1 + int(constraint_dim)
	if not diagnostics_history:
		return {
			"update_index": np.zeros((0,), dtype=np.int64),
			"global_step": np.zeros((0,), dtype=np.int64),
			"episode_index": np.zeros((0,), dtype=np.int64),
			"q_mean": np.zeros((0, head_count), dtype=np.float64),
			"q_std": np.zeros((0, head_count), dtype=np.float64),
			"grad_a_norm": np.zeros((0, head_count), dtype=np.float64),
			"actor_grad_norm": np.zeros((0, head_count), dtype=np.float64),
			"constraint_to_objective_grad_norm_ratio": np.zeros((0, int(constraint_dim)), dtype=np.float64),
			"score_grad_norm": np.zeros((0, head_count), dtype=np.float64),
			"pathwise_grad_norm": np.zeros((0, head_count), dtype=np.float64),
			"combined_grad_norm": np.zeros((0, head_count), dtype=np.float64),
			"qprop_eta": np.zeros((0, head_count), dtype=np.float64),
			"qprop_covariance": np.zeros((0, head_count), dtype=np.float64),
			"score_signal_mean": np.zeros((0, head_count), dtype=np.float64),
			"score_signal_std": np.zeros((0, head_count), dtype=np.float64),
			"control_signal_mean": np.zeros((0, head_count), dtype=np.float64),
			"control_signal_std": np.zeros((0, head_count), dtype=np.float64),
			"qprop_critic_loss": np.zeros((0, head_count), dtype=np.float64),
			"qprop_critic_td_error_mean": np.zeros((0, head_count), dtype=np.float64),
			"qprop_critic_target_mean": np.zeros((0, head_count), dtype=np.float64),
			"qprop_critic_pred_mean": np.zeros((0, head_count), dtype=np.float64),
			"qprop_replay_batch_size": np.zeros((0,), dtype=np.float64),
			"qprop_control_source_code": np.zeros((0,), dtype=np.float64),
			"qprop_target_action_mode_code": np.zeros((0,), dtype=np.float64),
			"qprop_pathwise_grad_ratio": np.zeros((0, head_count), dtype=np.float64),
			"qprop_score_grad_ratio": np.zeros((0, head_count), dtype=np.float64),
		}
	return {
		"update_index": np.asarray([item["update_index"] for item in diagnostics_history], dtype=np.int64),
		"global_step": np.asarray([item["global_step"] for item in diagnostics_history], dtype=np.int64),
		"episode_index": np.asarray([item["episode_index"] for item in diagnostics_history], dtype=np.int64),
		"q_mean": np.asarray([item["q_mean"] for item in diagnostics_history], dtype=np.float64),
		"q_std": np.asarray([item["q_std"] for item in diagnostics_history], dtype=np.float64),
		"grad_a_norm": np.asarray([item["grad_a_norm"] for item in diagnostics_history], dtype=np.float64),
		"actor_grad_norm": np.asarray([item["actor_grad_norm"] for item in diagnostics_history], dtype=np.float64),
		"constraint_to_objective_grad_norm_ratio": np.asarray(
			[item["constraint_to_objective_grad_norm_ratio"] for item in diagnostics_history],
			dtype=np.float64,
		),
		"score_grad_norm": np.asarray([item["score_grad_norm"] for item in diagnostics_history], dtype=np.float64),
		"pathwise_grad_norm": np.asarray([item["pathwise_grad_norm"] for item in diagnostics_history], dtype=np.float64),
		"combined_grad_norm": np.asarray([item["combined_grad_norm"] for item in diagnostics_history], dtype=np.float64),
		"qprop_eta": np.asarray([item["qprop_eta"] for item in diagnostics_history], dtype=np.float64),
		"qprop_covariance": np.asarray([item["qprop_covariance"] for item in diagnostics_history], dtype=np.float64),
		"score_signal_mean": np.asarray([item["score_signal_mean"] for item in diagnostics_history], dtype=np.float64),
		"score_signal_std": np.asarray([item["score_signal_std"] for item in diagnostics_history], dtype=np.float64),
		"control_signal_mean": np.asarray([item["control_signal_mean"] for item in diagnostics_history], dtype=np.float64),
		"control_signal_std": np.asarray([item["control_signal_std"] for item in diagnostics_history], dtype=np.float64),
		"qprop_critic_loss": np.asarray([item["qprop_critic_loss"] for item in diagnostics_history], dtype=np.float64),
		"qprop_critic_td_error_mean": np.asarray([item["qprop_critic_td_error_mean"] for item in diagnostics_history], dtype=np.float64),
		"qprop_critic_target_mean": np.asarray([item["qprop_critic_target_mean"] for item in diagnostics_history], dtype=np.float64),
		"qprop_critic_pred_mean": np.asarray([item["qprop_critic_pred_mean"] for item in diagnostics_history], dtype=np.float64),
		"qprop_replay_batch_size": np.asarray([item["qprop_replay_batch_size"] for item in diagnostics_history], dtype=np.float64),
		"qprop_control_source_code": np.asarray([item["qprop_control_source_code"] for item in diagnostics_history], dtype=np.float64),
		"qprop_target_action_mode_code": np.asarray([item["qprop_target_action_mode_code"] for item in diagnostics_history], dtype=np.float64),
		"qprop_pathwise_grad_ratio": np.asarray([item["qprop_pathwise_grad_ratio"] for item in diagnostics_history], dtype=np.float64),
		"qprop_score_grad_ratio": np.asarray([item["qprop_score_grad_ratio"] for item in diagnostics_history], dtype=np.float64),
	}


def _freeze_critic_head_parameters(critic, constraint_dim):
	previous_states = []
	for head_idx in range(1 + constraint_dim):
		net = getattr(critic, "target_net{0}".format(int(head_idx)))
		for para in net.parameters():
			previous_states.append((para, para.requires_grad))
			para.requires_grad_(False)
	return previous_states


def _restore_parameter_grad_states(previous_states):
	for para, requires_grad in previous_states:
		para.requires_grad_(requires_grad)


def _critic_all_head_values(critic, state_batch_torch, action_batch_torch, constraint_dim):
	values = []
	for head_idx in range(1 + int(constraint_dim)):
		values.append(_critic_head_value(critic, state_batch_torch, action_batch_torch, head_idx).detach().view(-1))
	return torch.stack(values, dim=1)


def _preprocess_score_signal(example_name, q_values_torch, head_idx):
	q_values_torch = q_values_torch.detach()
	head_values = q_values_torch[:, int(head_idx)]
	centered = head_values - torch.mean(head_values)
	if "MIMO" in str(example_name):
		return centered.detach()
	objective_values = q_values_torch[:, 0]
	objective_std = torch.std(objective_values, unbiased=False) + 1e-6
	return (centered / objective_std).detach()


def _compute_conservative_qprop_eta(score_signal, control_signal):
	score_centered = score_signal.detach().view(-1) - torch.mean(score_signal.detach().view(-1))
	control_centered = control_signal.detach().view(-1) - torch.mean(control_signal.detach().view(-1))
	covariance = torch.mean(score_centered * control_centered)
	eta_value = 1.0 if float(covariance.item()) > 0.0 else 0.0
	eta = torch.tensor(eta_value, dtype=score_signal.dtype, device=score_signal.device)
	return eta, covariance.detach()


def _compute_taylor_control_signal(critic, state_batch_torch, action_batch_torch, mu_torch, head_idx):
	q_mu = _control_critic_head_value(critic, state_batch_torch, mu_torch, head_idx).view(-1)
	action_grad = torch.autograd.grad(
		q_mu.sum(),
		mu_torch,
		retain_graph=True,
		create_graph=False,
		allow_unused=True,
	)[0]
	if action_grad is None:
		action_grad = torch.zeros_like(mu_torch)
	control_signal = torch.sum(action_grad.detach() * (action_batch_torch.detach() - mu_torch.detach()), dim=1)
	return control_signal.detach(), q_mu, action_grad.detach()


def _extract_loss_gradient_norm(actor, loss, real_theta_dim, normalize_actor_gradient, update_log_std):
	_zero_actor_gradients(actor)
	loss.backward(retain_graph=True)
	grad_tmp = _extract_actor_gradient(
		actor,
		real_theta_dim,
		normalize_actor_gradient,
		update_log_std,
	)
	grad_norm = float(torch.linalg.norm(grad_tmp.detach()).item())
	_zero_actor_gradients(actor)
	return grad_norm


def _compute_qprop_conservative_gradient(
	actor,
	score_critic,
	control_critic,
	state_batch_torch,
	action_batch_torch,
	q_behavior_all_torch,
	head_idx,
	real_theta_dim,
	normalize_actor_gradient,
	update_log_std,
	return_diagnostics,
):
	score_signal = _preprocess_score_signal(score_critic.example_name, q_behavior_all_torch, head_idx)
	mu_torch = actor.mean_action_tensor(state_batch_torch)
	control_signal, q_mu, action_grad = _compute_taylor_control_signal(
		control_critic,
		state_batch_torch,
		action_batch_torch,
		mu_torch,
		head_idx,
	)
	eta, covariance = _compute_conservative_qprop_eta(score_signal, control_signal)
	log_prob = actor.evaluate_action(state_batch_torch, action_batch_torch)
	score_residual_loss = torch.mean(log_prob * (score_signal - eta * control_signal).detach())
	pathwise_control_loss = eta * torch.mean(q_mu)
	combined_loss = score_residual_loss + pathwise_control_loss

	score_grad_norm = 0.0
	pathwise_grad_norm = 0.0
	if return_diagnostics:
		score_grad_norm = _extract_loss_gradient_norm(
			actor,
			score_residual_loss,
			real_theta_dim,
			normalize_actor_gradient,
			update_log_std,
		)
		pathwise_grad_norm = _extract_loss_gradient_norm(
			actor,
			pathwise_control_loss,
			real_theta_dim,
			normalize_actor_gradient,
			update_log_std,
		)

	_zero_actor_gradients(actor)
	combined_loss.backward()
	actor_gradient = _extract_actor_gradient(
		actor,
		real_theta_dim,
		normalize_actor_gradient,
		update_log_std,
	)
	diagnostics = {
		"q_mean": float(torch.mean(q_behavior_all_torch[:, int(head_idx)].detach()).item()),
		"q_std": float(torch.std(q_behavior_all_torch[:, int(head_idx)].detach(), unbiased=False).item()),
		"grad_a_norm": float(torch.linalg.norm(action_grad.detach()).item()),
		"actor_grad_norm": float(torch.linalg.norm(actor_gradient.detach()).item()),
		"score_grad_norm": float(score_grad_norm),
		"pathwise_grad_norm": float(pathwise_grad_norm),
		"combined_grad_norm": float(torch.linalg.norm(actor_gradient.detach()).item()),
		"qprop_eta": float(eta.detach().item()),
		"qprop_covariance": float(covariance.detach().item()),
		"score_signal_mean": float(torch.mean(score_signal.detach()).item()),
		"score_signal_std": float(torch.std(score_signal.detach(), unbiased=False).item()),
		"control_signal_mean": float(torch.mean(control_signal.detach()).item()),
		"control_signal_std": float(torch.std(control_signal.detach(), unbiased=False).item()),
	}
	grad_norm_denominator = float(score_grad_norm) + float(pathwise_grad_norm) + 1e-12
	diagnostics["qprop_pathwise_grad_ratio"] = float(pathwise_grad_norm) / grad_norm_denominator
	diagnostics["qprop_score_grad_ratio"] = float(score_grad_norm) / grad_norm_denominator
	return actor_gradient, diagnostics


def _compute_pathwise_gradients(
	actor,
	critic,
	state_batch_torch,
	action_batch_torch,
	constraint_dim,
	real_theta_dim,
	policy_gradient_mode,
	normalize_actor_gradient,
	update_log_std,
	return_diagnostics=False,
	qprop_control_critic=None,
	qprop_control_source_code=0.0,
	qprop_critic_diagnostics=None,
):
	head_count = 1 + constraint_dim
	grad_tilda_torch = torch.zeros((head_count, real_theta_dim), dtype=torch.float, device=actor.device)
	diagnostics = _empty_pathwise_diagnostics(head_count, constraint_dim)
	control_critic = qprop_control_critic if qprop_control_critic is not None else critic
	critic_param_states = _freeze_critic_head_parameters(critic, constraint_dim)
	control_critic_param_states = []
	if policy_gradient_mode == "qprop_conservative" and control_critic is not critic:
		control_critic_param_states = _freeze_critic_head_parameters(control_critic, constraint_dim)
	try:
		q_behavior_all_torch = None
		if policy_gradient_mode == "qprop_conservative":
			q_behavior_all_torch = _critic_all_head_values(critic, state_batch_torch, action_batch_torch, constraint_dim)
			diagnostics["qprop_control_source_code"] = float(qprop_control_source_code)
			if qprop_critic_diagnostics is not None:
				for key, value in qprop_critic_diagnostics.items():
					diagnostics[key] = value
		for head_idx in range(head_count):
			_zero_actor_gradients(actor)
			if policy_gradient_mode == "qprop_conservative":
				actor_gradient, qprop_diagnostics = _compute_qprop_conservative_gradient(
					actor,
					critic,
					control_critic,
					state_batch_torch,
					action_batch_torch,
					q_behavior_all_torch,
					head_idx,
					real_theta_dim,
					normalize_actor_gradient,
					update_log_std,
					return_diagnostics,
				)
				grad_tilda_torch[head_idx] = actor_gradient
				if return_diagnostics:
					for key, value in qprop_diagnostics.items():
						diagnostics[key][head_idx] = value
				continue
			if policy_gradient_mode == "stochastic_pathwise":
				action_for_grad = actor.sample_action_tensor(
					state_batch_torch,
					reparameterized=True,
					use_mean=False,
					track_log_std_grad=update_log_std,
				)
			else:
				action_for_grad = actor.sample_action_tensor(
					state_batch_torch,
					reparameterized=False,
					use_mean=True,
					track_log_std_grad=update_log_std,
				)
			if return_diagnostics and action_for_grad.requires_grad:
				action_for_grad.retain_grad()
			head_value = _critic_head_value(critic, state_batch_torch, action_for_grad, head_idx)
			head_objective = torch.mean(head_value)
			if return_diagnostics:
				head_value_detached = head_value.detach().view(-1)
				diagnostics["q_mean"][head_idx] = float(torch.mean(head_value_detached).item())
				diagnostics["q_std"][head_idx] = float(torch.std(head_value_detached, unbiased=False).item())
			head_objective.backward()
			if return_diagnostics:
				if getattr(action_for_grad, "grad", None) is not None:
					diagnostics["grad_a_norm"][head_idx] = float(torch.linalg.norm(action_for_grad.grad.detach()).item())
			actor_gradient = _extract_actor_gradient(
				actor,
				real_theta_dim,
				normalize_actor_gradient,
				update_log_std,
			)
			grad_tilda_torch[head_idx] = actor_gradient
			if return_diagnostics:
				diagnostics["actor_grad_norm"][head_idx] = float(torch.linalg.norm(actor_gradient.detach()).item())
	finally:
		_restore_parameter_grad_states(control_critic_param_states)
		_restore_parameter_grad_states(critic_param_states)
	if return_diagnostics:
		objective_norm = diagnostics["actor_grad_norm"][0]
		diagnostics["constraint_to_objective_grad_norm_ratio"] = diagnostics["actor_grad_norm"][1:] / max(objective_norm, 1e-12)
		return grad_tilda_torch, diagnostics
	return grad_tilda_torch


def _save_sldac_pathwise_checkpoint(
	args,
	example_name,
	actor,
	seed,
	device,
	run_tag,
	episode_index_1based,
	log_index_0based,
	global_step_0based,
	reward_history,
	cost_history,
	state_dim,
	action_dim,
	constraint_dim,
	real_theta_dim,
	constr_lim,
	save_reason,
	checkpoint_interval_episodes,
):
	checkpoint_dir = _get_checkpoint_dir(args, example_name, run_tag, seed)
	os.makedirs(checkpoint_dir, exist_ok=True)
	if save_reason == "final" and (episode_index_1based % checkpoint_interval_episodes != 0):
		filename = "episode_{0:04d}_final.pt".format(int(episode_index_1based))
	else:
		filename = "episode_{0:04d}.pt".format(int(episode_index_1based))

	checkpoint = {
		"schema_version": CHECKPOINT_SCHEMA_VERSION,
		"algorithm": PATHWISE_ALGORITHM_NAME,
		"example_name": str(example_name),
		"run_tag": str(run_tag),
		"seed": int(seed),
		"device": str(device),
		"episode_index_1based": int(episode_index_1based),
		"log_index_0based": int(log_index_0based),
		"global_step_0based": int(global_step_0based),
		"save_reason": str(save_reason),
		"config": _collect_config_snapshot(args),
		"model": {
			"actor_state_dict": _cpu_state_dict(actor.net),
			"actor_log_std": actor.log_std.detach().cpu().clone(),
		},
		"stats": {
			"reward_history": [float(item) for item in reward_history],
			"cost_history": [float(item) for item in cost_history],
			"latest_reward_average": float(reward_history[-1]),
			"latest_cost_average": float(cost_history[-1]),
		},
		"shapes": {
			"state_dim": int(state_dim),
			"action_dim": int(action_dim),
			"constraint_dim": int(constraint_dim),
			"real_theta_dim": int(real_theta_dim),
			"constr_lim": np.asarray(constr_lim, dtype=np.float64).reshape(-1).tolist(),
		},
	}
	torch.save(checkpoint, os.path.join(checkpoint_dir, filename))


def SLDAC_Pathwise_main(args, example_name):
	seed = int(getattr(args, "seed", 0))
	np.random.seed(seed)
	torch.manual_seed(seed)
	device = str(getattr(args, "device", "cpu")).lower()
	if device == "cuda" and (not torch.cuda.is_available()):
		device = "cpu"

	T = args.T
	grad_T = args.grad_T
	num_new_data = args.num_new_data
	update_time_per_episode = args.update_time_per_episode
	MAX_STEPS = args.MAX_STEPS
	alpha_pow = args.alpha_pow
	beta_pow = args.beta_pow
	eta_pow = args.eta_pow
	gamma_pow_reward = args.gamma_pow_reward
	gamma_pow_cost = args.gamma_pow_cost
	tau_reward = args.tau_reward
	tau_cost = args.tau_cost
	Q_update_time = args.Q_update_time
	window = args.window
	run_tag = _get_run_tag(args)
	checkpoint_interval_episodes = max(1, int(getattr(args, "checkpoint_interval_episodes", 10)))
	save_final_checkpoint = _arg_to_bool(getattr(args, "save_final_checkpoint", True))
	save_diagnostics = _arg_to_bool(getattr(args, "save_diagnostics", True))
	policy_gradient_mode, behavior_policy_mode, normalize_actor_gradient, update_log_std, print_actor_grad_norm = _resolve_algorithm_modes(args)
	use_qprop_dedicated_critic = _arg_to_bool(getattr(args, "use_qprop_dedicated_critic", True))
	qprop_critic_update_steps = max(1, int(getattr(args, "qprop_critic_update_steps", 1)))
	qprop_replay_batch_size = _arg_to_optional_int(getattr(args, "qprop_replay_batch_size", None))
	qprop_target_action_mode = _validate_choice(
		"qprop_target_action_mode",
		getattr(args, "qprop_target_action_mode", "mean"),
		("mean",),
	)
	qprop_critic_lr_scale = float(getattr(args, "qprop_critic_lr_scale", 1.0))
	qprop_target_tau_reward = _arg_to_optional_float(getattr(args, "qprop_target_tau_reward", None))
	qprop_target_tau_cost = _arg_to_optional_float(getattr(args, "qprop_target_tau_cost", None))
	total_episodes = int(getattr(args, "episode", 0))
	if total_episodes <= 0:
		total_episodes = int(getattr(args, "num_update_time", 0) / max(update_time_per_episode, 1))

	reward_average_save = []
	cost_average_save = []
	diagnostics_history = []
	env, actor, state_dim, action_dim, constraint_dim, constr_lim = _build_scene(example_name, seed, device, grad_T)
	buffer = DataStorage(T, window, num_new_data, 1, state_dim, action_dim, constraint_dim)
	critic = Critic(example_name, grad_T, state_dim, action_dim, constraint_dim, Q_update_time, device)
	qprop_critic = None
	if policy_gradient_mode == "qprop_conservative" and use_qprop_dedicated_critic:
		qprop_critic = QPropCritic(
			example_name,
			state_dim,
			action_dim,
			constraint_dim,
			device,
			qprop_lr_scale=qprop_critic_lr_scale,
			qprop_target_tau_reward=qprop_target_tau_reward,
			qprop_target_tau_cost=qprop_target_tau_cost,
		)
	paras_torch, real_theta_dim = _flatten_actor_parameters(actor)
	func_value = np.zeros(constraint_dim + 1)
	grad = np.zeros((constraint_dim + 1, real_theta_dim))

	observation = env.reset()
	update_index = 0
	print_index = 0
	Q_update_index = 0
	for t in range(MAX_STEPS):
		state = observation
		action = _build_behavior_action(actor, state, behavior_policy_mode)
		observation, reward, done, info = env.step(action)
		next_state = observation
		costs = np.zeros(constraint_dim + 1)
		costs[0] = reward
		for k in range(1, constraint_dim + 1):
			costs[k] = (info.get('cost_' + str(k), info.get('cost', 0)) - constr_lim[k - 1])
		aver_cost = info.get('cost', 0) / constraint_dim
		aver_reward = reward
		buffer.store_experiences(state, action, costs, next_state, aver_reward, aver_cost)

		if t > 2 * T and ((t - 2 * T) % (num_new_data / Q_update_time) == 0):
			Q_update_index += 1
			alpha = 1 / ((update_index + 1) ** alpha_pow)
			beta = 1 / ((update_index + 1) ** beta_pow)
			eta = 1 / ((update_index + 1) ** eta_pow)
			if Q_update_index == Q_update_time:
				gamma_reward = 1 / ((update_index + 1) ** gamma_pow_reward)
				gamma_cost = 1 / ((update_index + 1) ** gamma_pow_cost)
			else:
				gamma_reward = 0
				gamma_cost = 0

			state_buffer, action_buffer, costs_buffer, next_state_buffer, aver_reward_buffer, aver_cost_buffer = buffer.take_experiences()
			func_value_tilda = np.mean(costs_buffer, axis=0)
			func_value = (1 - alpha) * func_value + alpha * func_value_tilda
			if (update_index % update_time_per_episode == 0) and (Q_update_index == 1):
				reward_average = float(np.mean(aver_reward_buffer))
				cost_average = float(np.mean(aver_cost_buffer))
				print(PATHWISE_ALGORITHM_NAME + "_EPISODE: ", print_index)
				print('reward_average: ', reward_average)
				print('cost_average: ', cost_average)
				reward_average_save.append(reward_average)
				cost_average_save.append(cost_average)
				current_episode = print_index + 1
				save_reason = None
				if current_episode % checkpoint_interval_episodes == 0:
					save_reason = "interval"
				if save_final_checkpoint and total_episodes > 0 and current_episode == total_episodes:
					save_reason = "final"
				if save_reason is not None:
					_save_sldac_pathwise_checkpoint(
						args=args,
						example_name=example_name,
						actor=actor,
						seed=seed,
						device=device,
						run_tag=run_tag,
						episode_index_1based=current_episode,
						log_index_0based=print_index,
						global_step_0based=t,
						reward_history=reward_average_save,
						cost_history=cost_average_save,
						state_dim=state_dim,
						action_dim=action_dim,
						constraint_dim=constraint_dim,
						real_theta_dim=real_theta_dim,
						constr_lim=constr_lim,
						save_reason=save_reason,
						checkpoint_interval_episodes=checkpoint_interval_episodes,
					)
				print_index += 1

			state_batch = state_buffer[(2 * T - grad_T):2 * T]
			action_batch = action_buffer[(2 * T - grad_T):2 * T]
			costs_batch = costs_buffer[(2 * T - grad_T):2 * T]
			next_state_batch = next_state_buffer[(2 * T - grad_T):2 * T]
			next_action_batch = _build_behavior_action_batch(
				actor,
				next_state_buffer[(2 * T - grad_T):2 * T],
				action_dim,
				behavior_policy_mode,
			)
			state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=device)
			action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)
			critic.critic_update(func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost)

			if Q_update_index == Q_update_time:
				qprop_critic_diagnostics = None
				if qprop_critic is not None:
					qprop_critic_diagnostics = qprop_critic.update_from_replay(
						func_value=func_value,
						state_buffer=state_buffer,
						action_buffer=action_buffer,
						costs_buffer=costs_buffer,
						next_state_buffer=next_state_buffer,
						actor=actor,
						batch_size=grad_T if qprop_replay_batch_size is None else qprop_replay_batch_size,
						update_steps=qprop_critic_update_steps,
						target_action_mode=qprop_target_action_mode,
						tau_reward=gamma_reward if qprop_target_tau_reward is None else qprop_target_tau_reward,
						tau_cost=gamma_cost if qprop_target_tau_cost is None else qprop_target_tau_cost,
						rng=np.random,
					)
				update_index += 1
				Q_update_index = 0
				pathwise_gradient_result = _compute_pathwise_gradients(
					actor,
					critic,
					state_batch_torch,
					action_batch_torch,
					constraint_dim,
					real_theta_dim,
					policy_gradient_mode,
					normalize_actor_gradient,
					update_log_std,
					return_diagnostics=save_diagnostics,
					qprop_control_critic=qprop_critic,
					qprop_control_source_code=1.0 if qprop_critic is not None else 0.0,
					qprop_critic_diagnostics=qprop_critic_diagnostics,
				)
				if save_diagnostics:
					grad_tilda_torch, pathwise_diagnostics = pathwise_gradient_result
					pathwise_diagnostics["update_index"] = int(update_index)
					pathwise_diagnostics["global_step"] = int(t)
					pathwise_diagnostics["episode_index"] = int(print_index)
					diagnostics_history.append(pathwise_diagnostics)
				else:
					grad_tilda_torch = pathwise_gradient_result
				grad = (1 - alpha) * grad + alpha * grad_tilda_torch.detach().cpu().numpy()
				grad_norms = np.linalg.norm(grad_tilda_torch.detach().cpu().numpy(), axis=1)
				if print_actor_grad_norm:
					print('actor_grad_norms: ', grad_norms.tolist())

				paras_bar = update_policy(func_value, grad, paras_torch.detach().cpu().numpy(), tau_reward=tau_reward, tau_cost=tau_cost)
				paras_bar_torch = torch.tensor(paras_bar, dtype=torch.float, device=device)
				paras_torch = (1 - beta) * paras_torch + beta * paras_bar_torch
				_load_actor_parameters(actor, paras_torch, update_log_std)
				if not update_log_std:
					paras_torch, _ = _flatten_actor_parameters(actor)

	return reward_average_save, cost_average_save, _pack_pathwise_diagnostics(diagnostics_history, constraint_dim)
