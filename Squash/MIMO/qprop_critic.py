import numpy as np
import torch
import torch.nn.functional as F

from model import Critic_net_CLQR
from model import Critic_net_MIMO
from utils import hard_update
from utils import soft_update


QPROP_CONTROL_SOURCE_DEDICATED = 1.0
QPROP_TARGET_ACTION_MEAN = 0.0


def _flatten_network_params(modules):
	params = []
	for module in modules:
		for para in module.parameters():
			params.append(para.detach().view(-1).cpu())
	if not params:
		return np.zeros((0,), dtype=np.float64)
	return torch.cat(params).numpy().astype(np.float64, copy=False)


class QPropCritic:
	def __init__(
		self,
		example_name,
		state_dim,
		action_dim,
		constraint_dim,
		device,
		qprop_lr_scale=1.0,
		qprop_target_tau_reward=None,
		qprop_target_tau_cost=None,
	):
		self.example_name = str(example_name)
		self.state_dim = int(state_dim)
		self.action_dim = int(action_dim)
		self.constraint_dim = int(constraint_dim)
		self.head_count = 1 + self.constraint_dim
		self.device = device
		self.qprop_lr_scale = float(qprop_lr_scale)
		self.qprop_target_tau_reward = qprop_target_tau_reward
		self.qprop_target_tau_cost = qprop_target_tau_cost
		self._validate_head_count()
		self._build_heads()

	def _validate_head_count(self):
		if "MIMO" in self.example_name:
			if self.constraint_dim != 4:
				raise ValueError("MIMO QPropCritic requires constraint_dim == 4, got {0}".format(self.constraint_dim))
		else:
			if self.constraint_dim != 1:
				raise ValueError("CLQR QPropCritic requires constraint_dim == 1, got {0}".format(self.constraint_dim))

	def _make_head_net(self, head_idx):
		if "MIMO" in self.example_name:
			return Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
		return Critic_net_CLQR(self.state_dim, self.action_dim, self.device)

	def _head_base_lr(self, head_idx):
		if "MIMO" in self.example_name:
			return 0.1 * self.qprop_lr_scale
		if int(head_idx) == 0:
			return 0.1 * self.qprop_lr_scale
		return 0.005 * self.qprop_lr_scale

	def _build_heads(self):
		self._nets = []
		self._target_nets = []
		self._optimizers = []
		for head_idx in range(self.head_count):
			net = self._make_head_net(head_idx)
			target_net = self._make_head_net(head_idx)
			optimizer = torch.optim.Adam(net.parameters(), lr=self._head_base_lr(head_idx))
			hard_update(target_net, net)
			setattr(self, "net{0}".format(head_idx), net)
			setattr(self, "target_net{0}".format(head_idx), target_net)
			setattr(self, "critic{0}_optimizer".format(head_idx), optimizer)
			self._nets.append(net)
			self._target_nets.append(target_net)
			self._optimizers.append(optimizer)

	def _select_batch_indices(self, entry_count, batch_size, rng):
		entry_count = int(entry_count)
		if entry_count <= 0:
			raise ValueError("QPropCritic replay buffer is empty.")
		if batch_size is None:
			actual_batch_size = entry_count
		else:
			actual_batch_size = min(max(1, int(batch_size)), entry_count)
		if actual_batch_size >= entry_count:
			return np.arange(entry_count), actual_batch_size
		random_source = np.random if rng is None else rng
		return random_source.choice(entry_count, size=actual_batch_size, replace=False), actual_batch_size

	def _target_action(self, actor, next_state_torch, target_action_mode):
		if str(target_action_mode) != "mean":
			raise ValueError("qprop_target_action_mode must be 'mean', got {0!r}".format(target_action_mode))
		with torch.no_grad():
			return actor.mean_action_tensor(next_state_torch).detach()

	def _empty_update_diagnostics(self, actual_batch_size):
		return {
			"qprop_critic_loss": np.zeros(self.head_count, dtype=np.float64),
			"qprop_critic_td_error_mean": np.zeros(self.head_count, dtype=np.float64),
			"qprop_critic_target_mean": np.zeros(self.head_count, dtype=np.float64),
			"qprop_critic_pred_mean": np.zeros(self.head_count, dtype=np.float64),
			"qprop_replay_batch_size": float(actual_batch_size),
			"qprop_control_source_code": float(QPROP_CONTROL_SOURCE_DEDICATED),
			"qprop_target_action_mode_code": float(QPROP_TARGET_ACTION_MEAN),
		}

	def update_from_replay(
		self,
		func_value,
		state_buffer,
		action_buffer,
		costs_buffer,
		next_state_buffer,
		actor,
		batch_size,
		update_steps,
		target_action_mode,
		tau_reward,
		tau_cost,
		rng=None,
	):
		state_array = np.asarray(state_buffer, dtype=np.float64)
		action_array = np.asarray(action_buffer, dtype=np.float64)
		costs_array = np.asarray(costs_buffer, dtype=np.float64)
		next_state_array = np.asarray(next_state_buffer, dtype=np.float64)
		entry_count = min(state_array.shape[0], action_array.shape[0], costs_array.shape[0], next_state_array.shape[0])
		_, actual_batch_size = self._select_batch_indices(entry_count, batch_size, rng)
		update_steps = max(1, int(update_steps))
		diagnostics = self._empty_update_diagnostics(actual_batch_size)

		for _ in range(update_steps):
			indices, actual_batch_size = self._select_batch_indices(entry_count, batch_size, rng)
			diagnostics["qprop_replay_batch_size"] = float(actual_batch_size)
			state_torch = torch.tensor(state_array[indices], dtype=torch.float, device=self.device)
			action_torch = torch.tensor(action_array[indices], dtype=torch.float, device=self.device)
			costs_torch = torch.tensor(costs_array[indices], dtype=torch.float, device=self.device)
			next_state_torch = torch.tensor(next_state_array[indices], dtype=torch.float, device=self.device)
			func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
			next_action_torch = self._target_action(actor, next_state_torch, target_action_mode)

			for head_idx in range(self.head_count):
				net = self._nets[head_idx]
				target_net = self._target_nets[head_idx]
				optimizer = self._optimizers[head_idx]
				with torch.no_grad():
					next_val = target_net.forward(next_state_torch, next_action_torch).view(-1)
					y_expected = costs_torch[:, head_idx] - func_value_torch[head_idx] + next_val
				y_predicted = net.forward(state_torch, action_torch).view(-1)
				loss = F.smooth_l1_loss(y_predicted, y_expected)
				optimizer.zero_grad()
				loss.backward()
				optimizer.step()

				td_error = y_predicted.detach() - y_expected.detach()
				diagnostics["qprop_critic_loss"][head_idx] += float(loss.detach().item())
				diagnostics["qprop_critic_td_error_mean"][head_idx] += float(torch.mean(td_error).item())
				diagnostics["qprop_critic_target_mean"][head_idx] += float(torch.mean(y_expected.detach()).item())
				diagnostics["qprop_critic_pred_mean"][head_idx] += float(torch.mean(y_predicted.detach()).item())

				tau = float(tau_reward) if head_idx == 0 else float(tau_cost)
				soft_update(target_net, net, tau)

		for key in (
			"qprop_critic_loss",
			"qprop_critic_td_error_mean",
			"qprop_critic_target_mean",
			"qprop_critic_pred_mean",
		):
			diagnostics[key] = diagnostics[key] / float(update_steps)
		return diagnostics

	def head_value(self, head_idx, state_batch_torch, action_batch_torch, use_target=True):
		head_idx = int(head_idx)
		net = self._target_nets[head_idx] if bool(use_target) else self._nets[head_idx]
		return net.forward(state_batch_torch, action_batch_torch).view(-1)

	def all_head_values(self, state_batch_torch, action_batch_torch, use_target=True):
		values = []
		for head_idx in range(self.head_count):
			values.append(self.head_value(head_idx, state_batch_torch, action_batch_torch, use_target=use_target).detach())
		return torch.stack(values, dim=1)

	def flatten_parameters(self, include_target=True):
		modules = list(self._nets)
		if include_target:
			modules.extend(self._target_nets)
		return _flatten_network_params(modules)
