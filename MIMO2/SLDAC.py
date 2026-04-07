from environment import Environment_MIMO
from environment import Environment_CLQR
from critic_opt import Critic
from utils import update_policy
from model import GaussianPolicy_MIMO
from model import GaussianPolicy_CLQR
from buffer import DataStorage
import os
import numpy as np
import torch


CHECKPOINT_SCHEMA_VERSION = 1
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
)


def _format_seed_dir(seed):
	return "seed_{0}".format(int(seed))


def _arg_to_bool(value):
	if isinstance(value, str):
		return value.strip().lower() not in {"", "0", "false", "no", "off"}
	return bool(value)


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
	root = getattr(args, "checkpoint_root", os.path.join("checkpoints", "SLDAC"))
	if not root:
		root = os.path.join("checkpoints", "SLDAC")
	root = str(root)
	if not os.path.isabs(root):
		root = os.path.join(os.getcwd(), root)
	return os.path.join(root, str(example_name), run_tag, _format_seed_dir(seed))


def _collect_config_snapshot(args):
	config = {}
	for field_name in CHECKPOINT_CONFIG_FIELDS:
		config[field_name] = _normalize_config_value(getattr(args, field_name, None))
	return config


def _save_sldac_checkpoint(
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
		"algorithm": "SLDAC",
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


def SLDAC_main(args, example_name):
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
	window=args.window
	run_tag = _get_run_tag(args)
	checkpoint_interval_episodes = max(1, int(getattr(args, "checkpoint_interval_episodes", 10)))
	save_final_checkpoint = _arg_to_bool(getattr(args, "save_final_checkpoint", True))
	total_episodes = int(getattr(args, "episode", 0))
	if total_episodes <= 0:
		total_episodes = int(getattr(args, "num_update_time", 0) / max(update_time_per_episode, 1))

	reward_average_save = []
	cost_average_save = []
	if 'MIMO' in example_name:
		Nt, UE_num = 8, 4  # The number of antennas and users.
		state_dim = 2 * UE_num * Nt + UE_num
		action_dim = UE_num + 1
		env = Environment_MIMO(seed=seed, Nt=Nt, UE_num=UE_num)
		constraint_dim = UE_num
		constr_lim = [1.2, 1.2, 1.2, 1.2]
		actor = GaussianPolicy_MIMO(state_dim, action_dim, device, grad_T)
	else:
		state_dim, action_dim = 15, 4
		env = Environment_CLQR(seed=seed, state_dim=state_dim, action_dim=action_dim)
		constraint_dim = 1
		constr_lim = 380 * np.ones(constraint_dim)
		actor = GaussianPolicy_CLQR(state_dim, action_dim, device, grad_T)

	buffer = DataStorage(T, num_new_data, state_dim, action_dim, constraint_dim, window, 1)
	critic = Critic(example_name, grad_T, state_dim, action_dim, constraint_dim, Q_update_time, device)

	theta_dim = 0
	for para in actor.net.parameters():
		theta_dim += para.numel()
	real_theta_dim = theta_dim + action_dim  # the dimension of the policy parameter.
	# real_theta_dim = theta_dim  # use this when using the Beta policy
	paras_torch = torch.zeros((real_theta_dim,), dtype=torch.float, device=device)
	ind = 0
	for para in actor.net.parameters():
		tmp = para.numel()
		paras_torch[ind: ind + tmp] = para.data.view(-1)
		ind = ind + tmp
	paras_torch[ind:] = actor.log_std  # comment this when using the Beta policy
	func_value = np.zeros(constraint_dim + 1)
	grad = np.zeros((constraint_dim + 1, real_theta_dim))

	observation = env.reset()
	update_index = 0
	print_index = 0
	Q_update_index = 0
	for t in range(MAX_STEPS):
		# generate new data (sample one step of the env)
		state = observation
		action = actor.sample_action(state)
		observation, reward, done, info = env.step(action)  # reward is the objective cost in the paper.
		next_state = observation
		costs = np.zeros(constraint_dim + 1)
		costs[0] = reward
		for k in range(1, constraint_dim + 1):
			costs[k] = (info.get('cost_' + str(k), info.get('cost', 0)) - constr_lim[k - 1])

		aver_reward = reward
		aver_cost = info.get('cost', 0) / constraint_dim
		buffer.store_experiences(state, action, costs, next_state, aver_reward, aver_cost)

		# update the policy
		if t > 2*T and ((t-2*T) % (num_new_data/Q_update_time) == 0):
			Q_update_index += 1
			alpha = 1 / ((update_index+1) ** alpha_pow)
			beta = 1 / ((update_index+1) ** beta_pow)
			eta = 1 / ((update_index+1) ** eta_pow)
			if Q_update_index==Q_update_time:
				gamma_reward = 1 / ((update_index+1) ** gamma_pow_reward)
				gamma_cost = 1 / ((update_index+1) ** gamma_pow_cost)
			else:
				gamma_reward = 0
				gamma_cost = 0

			state_buffer, action_buffer, costs_buffer, next_state_buffer, aver_reward_batch, aver_cost_batch = buffer.take_experiences()
			func_value_tilda = np.mean(costs_buffer, axis=0)
			func_value = (1 - alpha) * func_value + alpha * func_value_tilda
			if (update_index % update_time_per_episode == 0) and (Q_update_index == 1):
				reward_average = float(np.mean(aver_reward_batch))
				cost_average = float(np.mean(aver_cost_batch))
				print('SLDAC_EPISODE: ', print_index)
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
					_save_sldac_checkpoint(
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
			next_action_batch = np.zeros((grad_T, action_dim))
			for jjj in range(grad_T):
				next_action_batch[jjj, :] = actor.sample_action(next_state_buffer[(2 * T - grad_T) + jjj, :])
			state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=device)
			action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)
			# estimate the Q-value
			critic.critic_update(func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost)

			if (Q_update_index == Q_update_time):
				# estimate the gradient
				update_index += 1
				Q_update_index = 0
				Q_hat_torch = critic.critic_value(state_batch_torch, action_batch_torch)
				Q_hat = Q_hat_torch.detach().cpu().numpy()
				Q_hat[:, 0] = (Q_hat[:, 0] - np.mean(Q_hat[:, 0])) / (np.std(Q_hat[:, 0]) + 1e-6)
				for _ in range(1, 1 + constraint_dim):
					Q_hat[:, _] = (Q_hat[:, _] - np.mean(Q_hat[:, _])) / (np.std(Q_hat[:, 0]) + 1e-6)
				Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=device)
				state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=device)
				action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=device)
				grad_tilda_torch = torch.zeros((1 + constraint_dim, real_theta_dim), dtype=torch.float, device=device)
				for _ in range(1 + constraint_dim):
					# calculate the gradient
					actor.zero_grad()
					log_prob = actor.evaluate_action(state_batch_torch, action_batch_torch)
					actor_loss = (Q_hat_torch[:, _] * log_prob).mean()
					actor_loss.backward()
					grad_tmp = torch.zeros(real_theta_dim, dtype=torch.float, device=device)
					ind = 0
					for para in actor.net.parameters():
						tmp = para.numel()
						grad_tmp[ind: ind + tmp] = para.grad.view(-1)
						ind = ind + tmp
					grad_tmp[ind:] = actor.log_std.grad  # comment this when using the Beta policy
					grad_tilda_torch[_] = grad_tmp
				grad = (1 - alpha) * grad + alpha * grad_tilda_torch.detach().cpu().numpy()

				# update the policy parameter
				paras_bar = update_policy(func_value, grad, paras_torch.detach().cpu().numpy(), tau_reward=tau_reward, tau_cost=tau_cost)
				paras_bar_torch = torch.tensor(paras_bar, dtype=torch.float, device=device)
				paras_torch = (1 - beta) * paras_torch + beta * paras_bar_torch
				ind = 0
				for para in actor.net.parameters():
					tmp = para.numel()
					para.data = paras_torch[ind: ind + tmp].view(para.shape)
					ind = ind + tmp
				actor.log_std = paras_torch[ind:]  # comment this when using the Beta policy

	return reward_average_save, cost_average_save
