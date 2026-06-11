import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 0.003
ACTOR_DISTRIBUTION = "legacy_bounded_mean_plain_gaussian"


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    value = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-value, value)


class CriticNetMIMO(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super(CriticNetMIMO, self).__init__()
        self.fcs1 = nn.Linear(state_dim, 256)
        self.fcs1.weight.data = fanin_init(self.fcs1.weight.data.size())
        self.fcs2 = nn.Linear(256, 128)
        self.fcs2.weight.data = fanin_init(self.fcs2.weight.data.size())
        self.fca1 = nn.Linear(action_dim, 128)
        self.fca1.weight.data = fanin_init(self.fca1.weight.data.size())
        self.fc2 = nn.Linear(256, 128)
        self.fc2.weight.data = fanin_init(self.fc2.weight.data.size())
        self.fc3 = nn.Linear(128, 1)
        self.fc3.weight.data.uniform_(-EPS, EPS)
        self.device = device
        self.to(self.device)

    def forward(self, state, action):
        if state.dim() == 1:
            state = state.view(1, -1)
        if action.dim() == 1:
            action = action.view(1, -1)
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x1 = torch.cat((s2, a1), dim=1)
        x2 = F.relu(self.fc2(x1))
        return 10 * torch.tanh(0.001 * self.fc3(x2))


class MLPGaussianMIMO(nn.Module):
    def __init__(self, state_dim, fc1_dim, fc2_dim, action_dim, device):
        super(MLPGaussianMIMO, self).__init__()
        self.fc1 = nn.Linear(state_dim, fc1_dim)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)
        self.fc2 = nn.Linear(fc1_dim, fc2_dim)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)
        self.fc3 = nn.Linear(fc2_dim, action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        if state.dim() == 1:
            state = state.view(1, -1)
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return 2.5 * torch.sigmoid(self.fc3(x))


class GaussianPolicy_MIMO(nn.Module):
    def __init__(self, state_dim, action_dim, device, num_new_data):
        super(GaussianPolicy_MIMO, self).__init__()
        self.fc1_dim = 128
        self.fc2_dim = 128
        self.net = MLPGaussianMIMO(state_dim, self.fc1_dim, self.fc2_dim, action_dim, device)
        self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
        self.action_dim = int(action_dim)
        self.num_new_data = int(num_new_data)
        self.device = device
        self.actor_distribution = ACTOR_DISTRIBUTION
        self.to(self.device)

    def evaluate_action(self, state_torch, action_torch):
        self.net.train()
        mu = self.net(state_torch)
        self.log_std.requires_grad = True
        std_eval = torch.exp(self.log_std).view(1, -1).repeat(mu.shape[0], 1)
        gaussian = torch.distributions.normal.Normal(mu, std_eval)
        return gaussian.log_prob(action_torch).sum(dim=1)

    def sample_action(self, state):
        self.net.eval()
        self.log_std.requires_grad = False
        state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            mu = self.net(state_torch)
            gaussian = torch.distributions.normal.Normal(mu, torch.exp(self.log_std))
            action = gaussian.sample()
        return action.detach().cpu().numpy().reshape(-1)


def actor_to_vector(actor):
    parts = [param.data.view(-1).detach().cpu() for param in actor.net.parameters()]
    parts.append(actor.log_std.detach().cpu().view(-1))
    return torch.cat(parts).numpy().astype(np.float64)


def vector_to_actor(actor, theta):
    theta_torch = torch.tensor(theta, dtype=torch.float, device=actor.device)
    index = 0
    for param in actor.net.parameters():
        count = param.numel()
        param.data = theta_torch[index : index + count].view(param.shape).clone()
        index += count
    actor.log_std = theta_torch[index : index + actor.action_dim].clone().detach()


def flatten_actor_grad(actor):
    grad_parts = []
    for param in actor.net.parameters():
        if param.grad is None:
            grad_parts.append(torch.zeros(param.numel(), dtype=torch.float, device=actor.device))
        else:
            grad_parts.append(param.grad.view(-1))
    if actor.log_std.grad is None:
        grad_parts.append(torch.zeros(actor.action_dim, dtype=torch.float, device=actor.device))
    else:
        grad_parts.append(actor.log_std.grad.view(-1))
    return torch.cat(grad_parts).detach().cpu().numpy().astype(np.float64)
