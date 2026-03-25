from environment import Environment_MIMO
from environment import Environment_CLQR
from critic_opt import Critic
from utils import update_policy
from model import GaussianPolicy_MIMO
from model import GaussianPolicy_CLQR
from buffer import DataStorage
import numpy as np
import torch


def SLDAC_main(args, example_name):
	seed = 0
	np.random.seed(seed)
	torch.manual_seed(seed)
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
				print('SCAOPO_EPISODE: ', print_index)
				print('reward_average: ', np.mean(aver_reward_batch))
				print('cost_average: ', np.mean(aver_cost_batch))
				reward_average_save.append(np.mean(aver_reward_batch))
				cost_average_save.append(np.mean(aver_cost_batch))
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


