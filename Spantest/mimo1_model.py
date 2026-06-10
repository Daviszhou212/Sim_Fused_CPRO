from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


EPS = 0.003


def fanin_init(size, fanin=None):
    fanin = fanin or size[0]
    v = 1.0 / np.sqrt(fanin)
    return torch.Tensor(size).uniform_(-v, v)


class CriticNetMIMO(nn.Module):
    def __init__(self, state_dim, action_dim, device):
        super().__init__()
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
        self.to(device)

    def forward(self, state, action):
        s1 = F.relu(self.fcs1(state))
        s2 = F.relu(self.fcs2(s1))
        a1 = F.relu(self.fca1(action))
        x = torch.cat((s2, a1), dim=1)
        x = F.relu(self.fc2(x))
        return 10 * torch.tanh(0.001 * self.fc3(x))


class MlpGaussianMIMO(nn.Module):
    def __init__(self, state_dim, hidden1, hidden2, action_dim, device):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden1)
        nn.init.orthogonal_(self.fc1.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc1.bias.data, 0.0)
        self.fc2 = nn.Linear(hidden1, hidden2)
        nn.init.orthogonal_(self.fc2.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc2.bias.data, 0.0)
        self.fc3 = nn.Linear(hidden2, action_dim)
        nn.init.orthogonal_(self.fc3.weight.data, gain=np.sqrt(2))
        nn.init.constant_(self.fc3.bias.data, 0.0)
        self.to(device)

    def forward(self, state):
        x = torch.tanh(self.fc1(state))
        x = torch.tanh(self.fc2(x))
        return 2.5 * torch.sigmoid(self.fc3(x))


class GaussianPolicyMIMO(nn.Module):
    def __init__(self, state_dim, action_dim, device, batch_size):
        super().__init__()
        self.net = MlpGaussianMIMO(state_dim, 128, 128, action_dim, device)
        self.log_std = -0.5 * torch.ones(action_dim, dtype=torch.float, device=device)
        self.action_dim = int(action_dim)
        self.batch_size = int(batch_size)
        self.device = device
        self.to(device)

    def evaluate_action(self, state_torch, action_torch):
        self.net.train()
        mean = self.net(state_torch)
        self.log_std.requires_grad = True
        std = torch.exp(self.log_std).view(1, -1).repeat(state_torch.shape[0], 1)
        return torch.distributions.normal.Normal(mean, std).log_prob(action_torch).sum(dim=1)

    def sample_action(self, state):
        self.net.eval()
        self.log_std.requires_grad = False
        state_torch = torch.tensor(state, dtype=torch.float, device=self.device)
        with torch.no_grad():
            mean = self.net(state_torch)
            std = torch.exp(self.log_std)
            action = torch.distributions.normal.Normal(mean, std).sample()
        return action.detach().cpu().numpy()

    def flat_parameters(self):
        params = [param.data.view(-1) for param in self.net.parameters()]
        params.append(self.log_std.data.view(-1))
        return torch.cat(params)

    def set_flat_parameters(self, flat_params):
        vector = flat_params.detach().to(self.device)
        index = 0
        for param in self.net.parameters():
            count = param.numel()
            param.data = vector[index : index + count].view(param.shape)
            index += count
        self.log_std = vector[index : index + self.action_dim].detach().clone()
        self.log_std.requires_grad_(True)

    def flat_gradient(self):
        grads = []
        for param in self.net.parameters():
            grads.append(param.grad.view(-1))
        grads.append(self.log_std.grad.view(-1))
        return torch.cat(grads)

