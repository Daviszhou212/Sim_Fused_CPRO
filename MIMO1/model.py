import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

EPS = 0.003

def fanin_init(size, fanin=None):
	fanin = fanin or size[0]
	v = 1. / np.sqrt(fanin)
	return torch.Tensor(size).uniform_(-v, v)


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
	def __init__(self, state_dim, action_dim, device, num_new_data):
		super(GaussianPolicy_MIMO, self).__init__()
		self.fc1_dim = 128
		self.fc2_dim = 128
		self.net = MLP_Gaussian_MIMO(state_dim, self.fc1_dim, self.fc2_dim, action_dim, device)
		self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
		self.action_dim = action_dim
		self.num_new_data = num_new_data
		self.device = device
		self.to(self.device)

	def forward(self, state, action):
		raise NotImplementedError

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		mu = self.net(state_torch)
		self.log_std.requires_grad = True
		self.std_eval = torch.exp(self.log_std)
		self.std_eval = self.std_eval.view(1, -1).repeat(self.num_new_data, 1)
		gaussian_ = torch.distributions.normal.Normal(mu, self.std_eval)
		log_prob_action = gaussian_.log_prob(action_torch).sum(dim=1)

		return log_prob_action


	def sample_action(self, state):
		self.net.eval()
		self.log_std.requires_grad = False
		state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
		with torch.no_grad():
			mu = self.net(state_torch)
			self.std_sample = torch.exp(self.log_std)
			gaussian_ = torch.distributions.normal.Normal(mu, self.std_sample)
			action = gaussian_.sample()

		return action.detach().cpu().numpy()
	
	
class MLP_Gaussian_MIMO(nn.Module):
	def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
		super(MLP_Gaussian_MIMO, self).__init__()
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

	def evaluate_action(self, state_torch, action_torch):
		self.net.train()
		mu = self.net(state_torch)
		self.log_std.requires_grad = True
		self.std_eval = torch.exp(self.log_std)
		self.std_eval = self.std_eval.view(1, -1).repeat(self.num_new_data, 1)
		gaussian_ = torch.distributions.normal.Normal(mu, self.std_eval)
		log_prob_action = gaussian_.log_prob(action_torch).sum(dim=1)

		return log_prob_action


	def sample_action(self, state):
		self.net.eval()
		self.log_std.requires_grad = False
		state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
		with torch.no_grad():
			mu = self.net(state_torch)
			self.std_sample = torch.exp(self.log_std)
			gaussian_ = torch.distributions.normal.Normal(mu, self.std_sample)
			action = gaussian_.sample()

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

