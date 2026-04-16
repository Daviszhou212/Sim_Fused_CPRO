import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003
# MIMO actor 默认隐藏层配置；长度表示层数，元素值表示各层宽度。
DEFAULT_MIMO_ACTOR_HIDDEN_DIMS = (128, 128)
# 旧 checkpoint 未记录结构时，按历史两层 128 口径解释。
LEGACY_MIMO_ACTOR_HIDDEN_DIMS = (128, 128)

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


def _expand_std(log_std, reference_torch):
	std = torch.exp(log_std)
	if reference_torch.dim() <= 1:
		return std
	return std.view(1, -1).expand(reference_torch.shape[0], -1)


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
		return self.net(state_torch)

	def sample_action_tensor(self, state_torch, reparameterized=False, use_mean=False, track_log_std_grad=True):
		self.net.train()
		self.log_std.requires_grad = bool(track_log_std_grad)
		mu = self.net(state_torch)
		if use_mean:
			return mu
		gaussian_ = torch.distributions.normal.Normal(mu, _expand_std(self.log_std, mu))
		if reparameterized:
			return gaussian_.rsample()
		return gaussian_.sample()

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		self.log_std.requires_grad = True
		mu = self.net(state_torch)
		gaussian_ = torch.distributions.normal.Normal(mu, _expand_std(self.log_std, mu))
		log_prob_action = gaussian_.log_prob(action_torch).sum(dim=1)

		return log_prob_action


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
		mu = 2.5 * torch.sigmoid(mu) # "if MIMO"
		return mu

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
		return self.net(state_torch)

	def sample_action_tensor(self, state_torch, reparameterized=False, use_mean=False, track_log_std_grad=True):
		self.net.train()
		self.log_std.requires_grad = bool(track_log_std_grad)
		mu = self.net(state_torch)
		if use_mean:
			return mu
		gaussian_ = torch.distributions.normal.Normal(mu, _expand_std(self.log_std, mu))
		if reparameterized:
			return gaussian_.rsample()
		return gaussian_.sample()

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		self.log_std.requires_grad = True
		mu = self.net(state_torch)
		gaussian_ = torch.distributions.normal.Normal(mu, _expand_std(self.log_std, mu))
		log_prob_action = gaussian_.log_prob(action_torch).sum(dim=1)

		return log_prob_action


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

