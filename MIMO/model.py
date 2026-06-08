import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003
# 动作变换的开区间下界；只用于策略分布，不复用初始化 EPS。
ACTION_EPS = 1e-6
# 反变换时的内部夹紧宽度，用于避免 logit/atanh/softplus inverse 的奇点。
ACTION_INVERSE_EPS = 1e-6
# MIMO power 维的动作上界。
MIMO_POWER_MAX = 2.5
# CLQR 动作的对称裁剪边界。
CLQR_ACTION_MAX = 1.5
# 当前 actor 概率分布版本，用于 checkpoint 记录与旧 checkpoint warning。
ACTOR_DISTRIBUTION = "squashed_gaussian_v1"
# MIMO actor 默认隐藏层配置；长度表示层数，元素值表示各层宽度。
DEFAULT_MIMO_ACTOR_HIDDEN_DIMS = (128, 128)
# 旧 checkpoint 未记录结构时，按历史两层 128 口径解释。
LEGACY_MIMO_ACTOR_HIDDEN_DIMS = (128, 128)
# CTDE 多小区 MIMO 的共享本地 actor 隐藏层配置；所有小区复用同一组参数。
DEFAULT_CTDE_MIMO_ACTOR_HIDDEN_DIMS = (128, 128)

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


def _expand_std(log_std, reference_torch):
	std = torch.exp(log_std)
	if reference_torch.dim() <= 1:
		return std
	return std.view(1, -1).expand(reference_torch.shape[0], -1)


def _as_feature_mask(reference_torch, indices):
	mask = torch.zeros(reference_torch.shape[-1], dtype=torch.bool, device=reference_torch.device)
	if indices:
		mask[list(indices)] = True
	view_shape = [1] * reference_torch.dim()
	view_shape[-1] = reference_torch.shape[-1]
	return mask.view(*view_shape)


def _default_mimo_reg_indices(action_dim):
	return (int(action_dim) - 1,)


def _softplus_inverse(positive_torch):
	x = positive_torch.clamp_min(ACTION_INVERSE_EPS)
	return x + torch.log(-torch.expm1(-x))


def mimo_transform_raw_action(raw_action_torch, reg_indices=None):
	reg_indices = _default_mimo_reg_indices(raw_action_torch.shape[-1]) if reg_indices is None else tuple(reg_indices)
	reg_mask = _as_feature_mask(raw_action_torch, reg_indices)
	power_action = ACTION_EPS + (MIMO_POWER_MAX - ACTION_EPS) * torch.sigmoid(raw_action_torch)
	reg_action = F.softplus(raw_action_torch) + ACTION_EPS
	return torch.where(reg_mask, reg_action, power_action)


def mimo_inverse_action_and_log_det(action_torch, reg_indices=None, clamp_to_support=False):
	reg_indices = _default_mimo_reg_indices(action_torch.shape[-1]) if reg_indices is None else tuple(reg_indices)
	reg_mask = _as_feature_mask(action_torch, reg_indices)
	power_valid = (action_torch > ACTION_EPS) & (action_torch < MIMO_POWER_MAX)
	reg_valid = action_torch > ACTION_EPS
	valid_per_dim = torch.where(reg_mask, reg_valid, power_valid)
	valid = valid_per_dim.all(dim=-1)

	power_action = action_torch.clamp(
		min=ACTION_EPS + ACTION_INVERSE_EPS,
		max=MIMO_POWER_MAX - ACTION_INVERSE_EPS,
	)
	power_z = (power_action - ACTION_EPS) / (MIMO_POWER_MAX - ACTION_EPS)
	raw_power = torch.logit(power_z)
	raw_reg = _softplus_inverse(action_torch - ACTION_EPS)
	raw_action = torch.where(reg_mask, raw_reg, raw_power)

	power_log_det = (
		torch.log(torch.tensor(MIMO_POWER_MAX - ACTION_EPS, dtype=action_torch.dtype, device=action_torch.device))
		+ F.logsigmoid(raw_action)
		+ F.logsigmoid(-raw_action)
	)
	reg_log_det = -F.softplus(-raw_action)
	log_det = torch.where(reg_mask, reg_log_det, power_log_det).sum(dim=-1)
	if clamp_to_support:
		valid = torch.ones_like(valid, dtype=torch.bool)
	return raw_action, log_det, valid


def mimo_transformed_gaussian_log_prob(raw_loc_torch, log_std_torch, action_torch, reg_indices=None, clamp_to_support=False):
	raw_action, log_det, valid = mimo_inverse_action_and_log_det(
		action_torch,
		reg_indices=reg_indices,
		clamp_to_support=clamp_to_support,
	)
	dist = torch.distributions.normal.Normal(raw_loc_torch, _expand_std(log_std_torch, raw_loc_torch))
	log_prob = dist.log_prob(raw_action).sum(dim=-1) - log_det
	log_prob = torch.where(valid, log_prob, torch.full_like(log_prob, -torch.inf))
	if log_prob.dim() == 0:
		return log_prob.view(1)
	return log_prob


def clqr_transform_raw_action(raw_action_torch):
	return CLQR_ACTION_MAX * torch.tanh(raw_action_torch)


def clqr_inverse_action_and_log_det(action_torch, clamp_to_support=False):
	valid_per_dim = (action_torch > -CLQR_ACTION_MAX) & (action_torch < CLQR_ACTION_MAX)
	valid = valid_per_dim.all(dim=-1)
	scaled_action = (action_torch / CLQR_ACTION_MAX).clamp(
		min=-1.0 + ACTION_INVERSE_EPS,
		max=1.0 - ACTION_INVERSE_EPS,
	)
	raw_action = 0.5 * (torch.log1p(scaled_action) - torch.log1p(-scaled_action))
	log_det_per_dim = (
		torch.log(torch.tensor(CLQR_ACTION_MAX, dtype=action_torch.dtype, device=action_torch.device))
		+ 2.0 * (np.log(2.0) - raw_action - F.softplus(-2.0 * raw_action))
	)
	log_det = log_det_per_dim.sum(dim=-1)
	if clamp_to_support:
		valid = torch.ones_like(valid, dtype=torch.bool)
	return raw_action, log_det, valid


def clqr_transformed_gaussian_log_prob(raw_loc_torch, log_std_torch, action_torch, clamp_to_support=False):
	raw_action, log_det, valid = clqr_inverse_action_and_log_det(
		action_torch,
		clamp_to_support=clamp_to_support,
	)
	dist = torch.distributions.normal.Normal(raw_loc_torch, _expand_std(log_std_torch, raw_loc_torch))
	log_prob = dist.log_prob(raw_action).sum(dim=-1) - log_det
	log_prob = torch.where(valid, log_prob, torch.full_like(log_prob, -torch.inf))
	if log_prob.dim() == 0:
		return log_prob.view(1)
	return log_prob


def normalize_hidden_dims(hidden_dims, field_name="hidden_dims"):
	if hidden_dims is None:
		raise ValueError("{0} must not be None.".format(field_name))
	try:
		values = tuple(int(item) for item in hidden_dims)
	except TypeError:
		raise ValueError("{0} must be an iterable of positive integers.".format(field_name))
	if len(values) <= 0:
		raise ValueError("{0} must contain at least one hidden layer.".format(field_name))
	for value in values:
		if value <= 0:
			raise ValueError("{0} must contain only positive integers. got {1!r}".format(field_name, hidden_dims))
	return values


def get_mimo_actor_hidden_dims(hidden_dims=None):
	source = DEFAULT_MIMO_ACTOR_HIDDEN_DIMS if hidden_dims is None else hidden_dims
	return normalize_hidden_dims(source, "mimo_actor_hidden_dims")


def get_legacy_mimo_actor_hidden_dims():
	return normalize_hidden_dims(LEGACY_MIMO_ACTOR_HIDDEN_DIMS, "legacy_mimo_actor_hidden_dims")


def get_ctde_mimo_actor_hidden_dims(hidden_dims=None):
	source = DEFAULT_CTDE_MIMO_ACTOR_HIDDEN_DIMS if hidden_dims is None else hidden_dims
	return normalize_hidden_dims(source, "ctde_mimo_actor_hidden_dims")


def get_action_transform_metadata():
	return {
		"mimo_power": ["sigmoid_interval", ACTION_EPS, MIMO_POWER_MAX],
		"mimo_reg": ["softplus_positive", ACTION_EPS],
		"clqr": ["tanh_interval", -CLQR_ACTION_MAX, CLQR_ACTION_MAX],
	}


class Critic_net_MIMO(nn.Module):

	def __init__(self, state_dim, action_dim, device):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic_net_MIMO, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim, 64)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(64, 32)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim, 32)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(64, 32)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(32, 1)
		self.fc3.weight.data.uniform_(-EPS, EPS)
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x1 = torch.cat((s2, a1), dim=1)
		x2 = F.relu(self.fc2(x1))
		x3 = 10*torch.tanh(0.001*self.fc3(x2))
		return x3

class Critic_net_CLQR(nn.Module):

	def __init__(self, state_dim, action_dim, device):
		"""
		:param state_dim: Dimension of input state (int)
		:param action_dim: Dimension of input action (int)
		:return:
		"""
		super(Critic_net_CLQR, self).__init__()

		self.state_dim = state_dim
		self.action_dim = action_dim

		self.fcs1 = nn.Linear(state_dim, 64)
		self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
		self.fcs2 = nn.Linear(64, 32)
		self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())

		self.fca1 = nn.Linear(action_dim, 32)
		self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())

		self.fc2 = nn.Linear(64, 32)
		self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())

		self.fc3 = nn.Linear(32, 1)
		self.fc3.weight.data.uniform_(-EPS, EPS)
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		"""
		returns Value function Q(s,a) obtained from critic network
		:param state: Input state (Torch Variable : [n,state_dim] )
		:param action: Input Action (Torch Variable : [n,action_dim] )
		:return: Value function : Q(S,a) (Torch Variable : [n,1] )
		"""
		s1 = F.relu(self.fcs1(state))
		s2 = F.relu(self.fcs2(s1))
		a1 = F.relu(self.fca1(action))
		x1 = torch.cat((s2,a1),dim=1)
		x2 = F.relu(self.fc2(x1))
		x3 = 200*torch.tanh(self.fc3(x2))
		return x3


class GaussianPolicy_MIMO(nn.Module):
	"""The class to realize the Gaussian policy.
	The MIMO and CLQR have different bounds of action space. Thus some hyper-paras are different."""
	def __init__(self, state_dim, action_dim, device, num_new_data, hidden_dims=None):
		super(GaussianPolicy_MIMO, self).__init__()
		self.hidden_dims = get_mimo_actor_hidden_dims(hidden_dims)
		self.fc1_dim = int(self.hidden_dims[0])
		self.fc2_dim = int(self.hidden_dims[1]) if len(self.hidden_dims) > 1 else int(self.hidden_dims[0])
		self.net = MLP_Gaussian_MIMO(state_dim, self.hidden_dims, action_dim, device)
		self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
		self.action_dim = action_dim
		self.num_new_data = num_new_data
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		raise NotImplementedError

	def mean_action_tensor(self, state_torch):
		self.net.train()
		return mimo_transform_raw_action(self.net(state_torch), reg_indices=self._reg_indices())

	def evaluate_action_with_log_std(self, state_torch, action_torch, log_std_torch, clamp_to_support=False):
		self.net.train()
		return mimo_transformed_gaussian_log_prob(
			self.net(state_torch),
			log_std_torch,
			action_torch,
			reg_indices=self._reg_indices(),
			clamp_to_support=clamp_to_support,
		)

	def _reg_indices(self):
		return _default_mimo_reg_indices(self.action_dim)

	def sample_action_tensor(self, state_torch, reparameterized=False, use_mean=False, track_log_std_grad=True):
		self.net.train()
		self.log_std.requires_grad = bool(track_log_std_grad)
		raw_loc = self.net(state_torch)
		if use_mean:
			return mimo_transform_raw_action(raw_loc, reg_indices=self._reg_indices())
		gaussian_ = torch.distributions.normal.Normal(raw_loc, _expand_std(self.log_std, raw_loc))
		if reparameterized:
			raw_action = gaussian_.rsample()
		else:
			raw_action = gaussian_.sample()
		return mimo_transform_raw_action(raw_action, reg_indices=self._reg_indices())

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		self.log_std.requires_grad = True
		return self.evaluate_action_with_log_std(state_torch, action_torch, self.log_std)


	def sample_action(self, state, use_mean=False):
		self.net.eval()
		self.log_std.requires_grad = False
		state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
		with torch.no_grad():
			action = self.sample_action_tensor(
				state_torch,
				reparameterized=False,
				use_mean=use_mean,
				track_log_std_grad=False,
			)

		return action.detach().cpu().numpy()
	
	
class MLP_Gaussian_MIMO(nn.Module):
	def __init__(self, state_dim, hidden_dims, action_dim, device):
		super(MLP_Gaussian_MIMO, self).__init__()
		self.input_dim = state_dim
		self.hidden_dims = get_mimo_actor_hidden_dims(hidden_dims)
		self.fc1_dim = int(self.hidden_dims[0])
		self.fc2_dim = int(self.hidden_dims[1]) if len(self.hidden_dims) > 1 else int(self.hidden_dims[0])
		self.action_dim = action_dim
		self.hidden_layer_names = []

		prev_dim = self.input_dim
		for layer_idx, hidden_dim in enumerate(self.hidden_dims, start=1):
			layer = nn.Linear(prev_dim, int(hidden_dim))
			nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2))
			nn.init.constant_(layer.bias.data, 0.0)
			layer_name = "fc{0}".format(int(layer_idx))
			setattr(self, layer_name, layer)
			self.hidden_layer_names.append(layer_name)
			prev_dim = int(hidden_dim)

		self.output_layer_name = "fc{0}".format(int(len(self.hidden_dims) + 1))
		output_layer = nn.Linear(prev_dim, self.action_dim)
		nn.init.orthogonal_(output_layer.weight.data, gain=np.sqrt(2))
		nn.init.constant_(output_layer.bias.data, 0.0)
		setattr(self, self.output_layer_name, output_layer)
		self.device = device
		self.to(self.device)

	def forward(self, state):
		x = state
		for layer_name in self.hidden_layer_names:
			x = getattr(self, layer_name)(x)
			x = torch.tanh(x)
		mu = getattr(self, self.output_layer_name)(x)
		return mu


class GaussianPolicy_MultiCellMIMO_CTDE(nn.Module):
	"""Centralized-training/decentralized-execution Gaussian policy."""
	def __init__(
		self,
		state_dim,
		action_dim,
		device,
		num_new_data,
		cell_num,
		user_per_cell,
		Nt,
		hidden_dims=None,
	):
		super(GaussianPolicy_MultiCellMIMO_CTDE, self).__init__()
		self.cell_num = int(cell_num)
		self.num_cells = self.cell_num
		self.user_per_cell = int(user_per_cell)
		self.users_per_cell = self.user_per_cell
		self.Nt = int(Nt)
		self.cell_action_dim = self.user_per_cell + 1
		self.local_state_dim = 2 * self.user_per_cell * self.Nt + self.user_per_cell
		self.hidden_dims = get_ctde_mimo_actor_hidden_dims(hidden_dims)
		self.fc1_dim = int(self.hidden_dims[0])
		self.fc2_dim = int(self.hidden_dims[1]) if len(self.hidden_dims) > 1 else int(self.hidden_dims[0])
		expected_action_dim = self.cell_num * self.cell_action_dim
		if int(action_dim) != expected_action_dim:
			raise ValueError(
				"CTDE MIMO action_dim mismatch: expected {0}, got {1}".format(
					expected_action_dim,
					int(action_dim),
				)
			)
		self.net = CTDE_MultiCell_MIMO_Net(
			state_dim,
			self.hidden_dims,
			self.cell_num,
			self.user_per_cell,
			self.Nt,
			device,
		)
		self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
		self.action_dim = int(action_dim)
		self.num_new_data = num_new_data
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		raise NotImplementedError

	def mean_action_tensor(self, state_torch):
		self.net.train()
		return mimo_transform_raw_action(self.net(state_torch), reg_indices=self._reg_indices())

	def evaluate_action_with_log_std(self, state_torch, action_torch, log_std_torch, clamp_to_support=False):
		self.net.train()
		return mimo_transformed_gaussian_log_prob(
			self.net(state_torch),
			log_std_torch,
			action_torch,
			reg_indices=self._reg_indices(),
			clamp_to_support=clamp_to_support,
		)

	def _reg_indices(self):
		return tuple(range(self.cell_action_dim - 1, self.action_dim, self.cell_action_dim))

	def _cell_reg_indices(self):
		return (self.cell_action_dim - 1,)

	def sample_action_tensor(self, state_torch, reparameterized=False, use_mean=False, track_log_std_grad=True):
		self.net.train()
		self.log_std.requires_grad = bool(track_log_std_grad)
		raw_loc = self.net(state_torch)
		if use_mean:
			return mimo_transform_raw_action(raw_loc, reg_indices=self._reg_indices())
		gaussian_ = torch.distributions.normal.Normal(raw_loc, _expand_std(self.log_std, raw_loc))
		if reparameterized:
			raw_action = gaussian_.rsample()
		else:
			raw_action = gaussian_.sample()
		return mimo_transform_raw_action(raw_action, reg_indices=self._reg_indices())

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		self.log_std.requires_grad = True
		return self.evaluate_action_with_log_std(state_torch, action_torch, self.log_std)

	def sample_action(self, state, use_mean=False):
		self.net.eval()
		self.log_std.requires_grad = False
		state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
		with torch.no_grad():
			action = self.sample_action_tensor(
				state_torch,
				reparameterized=False,
				use_mean=use_mean,
				track_log_std_grad=False,
			)
		return action.detach().cpu().numpy()

	def mean_cell_action_tensor(self, local_state_torch):
		self.net.train()
		return mimo_transform_raw_action(self.net.local_forward(local_state_torch), reg_indices=self._cell_reg_indices())

	def sample_cell_action(self, local_state, cell_index=0, use_mean=True):
		# 分散执行接口：单个小区只依赖本地 CSI 和本地 delay。
		self.net.eval()
		local_state_torch = torch.tensor(local_state, dtype=torch.float, device=self.device)
		with torch.no_grad():
			raw_loc = self.net.local_forward(local_state_torch)
			if use_mean:
				action = mimo_transform_raw_action(raw_loc, reg_indices=self._cell_reg_indices())
			else:
				cell = int(cell_index)
				if cell < 0 or cell >= self.cell_num:
					raise ValueError("cell_index out of range: {0}".format(cell_index))
				start = cell * self.cell_action_dim
				end = start + self.cell_action_dim
				std = torch.exp(self.log_std.detach()[start:end])
				if raw_loc.dim() > 1:
					std = std.view(1, -1).expand(raw_loc.shape[0], -1)
				raw_action = torch.distributions.normal.Normal(raw_loc, std).sample()
				action = mimo_transform_raw_action(raw_action, reg_indices=self._cell_reg_indices())
		return action.detach().cpu().numpy()


class CTDE_MultiCell_MIMO_Net(nn.Module):
	def __init__(self, state_dim, hidden_dims, cell_num, user_per_cell, Nt, device):
		super(CTDE_MultiCell_MIMO_Net, self).__init__()
		self.input_dim = int(state_dim)
		self.hidden_dims = get_ctde_mimo_actor_hidden_dims(hidden_dims)
		self.cell_num = int(cell_num)
		self.user_per_cell = int(user_per_cell)
		self.Nt = int(Nt)
		self.local_state_dim = 2 * self.user_per_cell * self.Nt + self.user_per_cell
		self.cell_action_dim = self.user_per_cell + 1
		self.action_dim = self.cell_num * self.cell_action_dim
		self.channel_size = self.cell_num * self.user_per_cell * self.cell_num * self.Nt
		expected_state_dim = 2 * self.channel_size + self.cell_num * self.user_per_cell
		if self.input_dim != expected_state_dim:
			raise ValueError(
				"CTDE MIMO state_dim mismatch: expected {0}, got {1}".format(
					expected_state_dim,
					self.input_dim,
				)
			)

		self.hidden_layer_names = []
		prev_dim = self.local_state_dim
		for layer_idx, hidden_dim in enumerate(self.hidden_dims, start=1):
			layer = nn.Linear(prev_dim, int(hidden_dim))
			nn.init.orthogonal_(layer.weight.data, gain=np.sqrt(2))
			nn.init.constant_(layer.bias.data, 0.0)
			layer_name = "local_fc{0}".format(int(layer_idx))
			setattr(self, layer_name, layer)
			self.hidden_layer_names.append(layer_name)
			prev_dim = int(hidden_dim)

		self.output_layer_name = "local_fc{0}".format(int(len(self.hidden_dims) + 1))
		output_layer = nn.Linear(prev_dim, self.cell_action_dim)
		nn.init.orthogonal_(output_layer.weight.data, gain=np.sqrt(2))
		nn.init.constant_(output_layer.bias.data, 0.0)
		setattr(self, self.output_layer_name, output_layer)
		self.device = device
		self.to(self.device)

	def _extract_local_states(self, state):
		was_1d = state.dim() == 1
		if was_1d:
			state = state.view(1, -1)
		batch_size = int(state.shape[0])
		h_real = state[:, : self.channel_size].view(
			batch_size,
			self.cell_num,
			self.user_per_cell,
			self.cell_num,
			self.Nt,
		)
		h_imag = state[:, self.channel_size : 2 * self.channel_size].view(
			batch_size,
			self.cell_num,
			self.user_per_cell,
			self.cell_num,
			self.Nt,
		)
		delay = state[:, 2 * self.channel_size :].view(batch_size, self.cell_num, self.user_per_cell)
		local_blocks = []
		for cell in range(self.cell_num):
			local_blocks.append(
				torch.cat(
					(
						h_real[:, cell, :, cell, :].reshape(batch_size, -1),
						h_imag[:, cell, :, cell, :].reshape(batch_size, -1),
						delay[:, cell, :],
					),
					dim=1,
				)
			)
		local_state = torch.stack(local_blocks, dim=1)
		return local_state.reshape(batch_size * self.cell_num, self.local_state_dim), batch_size, was_1d

	def local_forward(self, local_state):
		was_1d = local_state.dim() == 1
		if was_1d:
			local_state = local_state.view(1, -1)
		x = local_state
		for layer_name in self.hidden_layer_names:
			x = getattr(self, layer_name)(x)
			x = torch.tanh(x)
		mu = getattr(self, self.output_layer_name)(x)
		if was_1d:
			return mu.reshape(-1)
		return mu

	def forward(self, state):
		local_state, batch_size, was_1d = self._extract_local_states(state)
		cell_mu = self.local_forward(local_state).view(batch_size, self.cell_num, self.cell_action_dim)
		joint_mu = cell_mu.reshape(batch_size, self.action_dim)
		if was_1d:
			return joint_mu.reshape(-1)
		return joint_mu


class GaussianPolicy_CLQR(nn.Module):
	"""The class to realize the Gaussian policy.
	The MIMO and CLQR have different bounds of action space. Thus some hyper-paras are different."""
	def __init__(self, state_dim, action_dim, device, num_new_data):
		super(GaussianPolicy_CLQR, self).__init__()
		self.fc1_dim = 128
		self.fc2_dim = 128
		self.net = MLP_Gaussian_CLQR(state_dim, self.fc1_dim, self.fc2_dim, action_dim, device)
		self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
		self.action_dim = action_dim
		self.num_new_data = num_new_data
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		raise NotImplementedError

	def mean_action_tensor(self, state_torch):
		self.net.train()
		return clqr_transform_raw_action(self.net(state_torch))

	def evaluate_action_with_log_std(self, state_torch, action_torch, log_std_torch, clamp_to_support=False):
		self.net.train()
		return clqr_transformed_gaussian_log_prob(
			self.net(state_torch),
			log_std_torch,
			action_torch,
			clamp_to_support=clamp_to_support,
		)

	def sample_action_tensor(self, state_torch, reparameterized=False, use_mean=False, track_log_std_grad=True):
		self.net.train()
		self.log_std.requires_grad = bool(track_log_std_grad)
		raw_loc = self.net(state_torch)
		if use_mean:
			return clqr_transform_raw_action(raw_loc)
		gaussian_ = torch.distributions.normal.Normal(raw_loc, _expand_std(self.log_std, raw_loc))
		if reparameterized:
			raw_action = gaussian_.rsample()
		else:
			raw_action = gaussian_.sample()
		return clqr_transform_raw_action(raw_action)

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		self.log_std.requires_grad = True
		return self.evaluate_action_with_log_std(state_torch, action_torch, self.log_std)


	def sample_action(self, state, use_mean=False):
		self.net.eval()
		self.log_std.requires_grad = False
		state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
		with torch.no_grad():
			action = self.sample_action_tensor(
				state_torch,
				reparameterized=False,
				use_mean=use_mean,
				track_log_std_grad=False,
			)

		return action.detach().cpu().numpy()


class MLP_Gaussian_CLQR(nn.Module):
	def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
		super(MLP_Gaussian_CLQR, self).__init__()
		self.input_dim = state_dim
		self.fc1_dim = fc1_dim
		self.fc2_dim = fc2_dim
		self.action_dim = action_dim

		self.fc1 = nn.Linear(self.input_dim, self.fc1_dim)
		nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
		nn.init.constant_(self.fc1.bias.data, 0.0)

		self.fc2 = nn.Linear(self.fc1_dim, self.fc2_dim)
		nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
		nn.init.constant_(self.fc2.bias.data, 0.0)

		self.fc3 = nn.Linear(self.fc2_dim, self.action_dim)
		nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
		nn.init.constant_(self.fc3.bias.data, 0.0)
		self.device = device
		self.to(self.device)

	def forward(self, state):
		x = self.fc1(state)
		x = torch.tanh(x)
		x = self.fc2(x)
		x = torch.tanh(x)
		mu = self.fc3(x)
		return mu

