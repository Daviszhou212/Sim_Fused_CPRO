from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from .mimo1_model import CriticNetMIMO


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


class Critic:
    def __init__(self, state_dim, action_dim, constraint_dim, q_update_time, device):
        self.constraint_dim = int(constraint_dim)
        self.device = device
        self.q_update_time = int(q_update_time)
        self.nets = []
        self.target_nets = []
        self.optimizers = []
        for _ in range(1 + self.constraint_dim):
            net = CriticNetMIMO(state_dim, action_dim, device)
            target = CriticNetMIMO(state_dim, action_dim, device)
            hard_update(target, net)
            self.nets.append(net)
            self.target_nets.append(target)
            self.optimizers.append(torch.optim.Adam(net.parameters(), 0.1 / np.sqrt(max(self.q_update_time, 1))))

    def update(self, func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost):
        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        state_t = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_t = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        costs_t = torch.tensor(costs_batch, dtype=torch.float, device=self.device)
        next_state_t = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        next_action_t = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)
        for head_idx, (net, target, optimizer) in enumerate(zip(self.nets, self.target_nets, self.optimizers)):
            next_val = torch.squeeze(target.forward(next_state_t, next_action_t).detach())
            y_expected = costs_t[:, head_idx] - func_value_torch[head_idx] + next_val
            y_predicted = torch.squeeze(net.forward(state_t, action_t))
            loss = F.smooth_l1_loss(y_predicted, y_expected)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            soft_update(target, net, gamma_reward if head_idx == 0 else gamma_cost)

    def value(self, state_batch_torch, action_batch_torch):
        q_hat = np.zeros((state_batch_torch.shape[0], 1 + self.constraint_dim))
        for head_idx, target in enumerate(self.target_nets):
            q_hat[:, head_idx] = target.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
        return torch.tensor(q_hat, dtype=torch.float, device=self.device)

