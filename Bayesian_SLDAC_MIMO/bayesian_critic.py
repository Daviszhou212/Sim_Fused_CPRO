import copy

import numpy as np
import torch
import torch.nn.functional as F

from .model import CriticNetMIMO


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def risk_correct_q_values(q_mean, q_std, beta_uncertainty):
    q_mean = np.asarray(q_mean, dtype=np.float64)
    q_std = np.asarray(q_std, dtype=np.float64)
    corrected = q_mean.copy()
    beta = float(beta_uncertainty)
    corrected[:, 0] = q_mean[:, 0] - beta * q_std[:, 0]
    if corrected.shape[1] > 1:
        corrected[:, 1:] = q_mean[:, 1:] + beta * q_std[:, 1:]
    return corrected


def normalize_q_hat_like_sldac_code(q_values):
    q_hat = np.asarray(q_values, dtype=np.float64).copy()
    q_hat[:, 0] = (q_hat[:, 0] - np.mean(q_hat[:, 0])) / (np.std(q_hat[:, 0]) + 1e-6)
    for head_idx in range(1, q_hat.shape[1]):
        q_hat[:, head_idx] = (q_hat[:, head_idx] - np.mean(q_hat[:, head_idx])) / (np.std(q_hat[:, 0]) + 1e-6)
    return q_hat


class BayesianCritic:
    def __init__(
        self,
        example_name,
        num_new_data,
        state_dim,
        action_dim,
        constraint_dim,
        q,
        device,
        ensemble_size=5,
        bootstrap_mask_prob=0.8,
        critic_seed=10000,
        ensemble_init_mode="shared",
        critic_lr_base=0.1,
    ):
        self.example_name = str(example_name)
        self.num_new_data = int(num_new_data)
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.constraint_dim = int(constraint_dim)
        self.q = int(q)
        self.device = device
        self.ensemble_size = int(ensemble_size)
        self.bootstrap_mask_prob = float(bootstrap_mask_prob)
        self.critic_seed = int(critic_seed)
        self.ensemble_init_mode = str(ensemble_init_mode)
        self.critic_lr_base = float(critic_lr_base)
        if self.ensemble_init_mode not in ("shared", "independent"):
            raise ValueError("ensemble_init_mode must be 'shared' or 'independent'")
        self.bootstrap_generator = torch.Generator(device="cpu")
        self.bootstrap_generator.manual_seed(self.critic_seed + 7919)
        self.head_count = 1 + self.constraint_dim
        self.head_members = []
        self.target_head_members = []
        self.optimizers = []
        base_lr = self.critic_lr_base / np.sqrt(max(self.q, 1))
        for _head in range(self.head_count):
            members = []
            targets = []
            opts = []
            shared_net = None
            shared_target = None
            for _member in range(self.ensemble_size):
                if self.ensemble_init_mode == "shared" and _member > 0:
                    net = copy.deepcopy(shared_net)
                    target = copy.deepcopy(shared_target)
                else:
                    net, target = self._make_member_pair(_head, _member)
                hard_update(target, net)
                if self.ensemble_init_mode == "shared" and _member == 0:
                    shared_net = net
                    shared_target = target
                members.append(net)
                targets.append(target)
                opts.append(torch.optim.Adam(net.parameters(), base_lr))
            self.head_members.append(members)
            self.target_head_members.append(targets)
            self.optimizers.append(opts)

    def _make_member_pair(self, head_idx, member_idx):
        if int(member_idx) == 0:
            return (
                CriticNetMIMO(self.state_dim, self.action_dim, self.device),
                CriticNetMIMO(self.state_dim, self.action_dim, self.device),
            )
        member_seed = self.critic_seed + 1009 * int(head_idx) + int(member_idx)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(member_seed)
            return (
                CriticNetMIMO(self.state_dim, self.action_dim, self.device),
                CriticNetMIMO(self.state_dim, self.action_dim, self.device),
            )

    def _td_loss(self, net, target, func_value_torch, head_idx, state_torch, action_torch, costs_torch, next_state_torch, next_action_torch):
        next_val = torch.squeeze(target.forward(next_state_torch, next_action_torch).detach())
        target_value = costs_torch[:, head_idx] - func_value_torch[head_idx] + next_val
        predicted = torch.squeeze(net.forward(state_torch, action_torch))
        return F.smooth_l1_loss(predicted, target_value, reduction="none")

    def critic_update(self, func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost):
        del eta
        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        state_torch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_torch = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        costs_torch = torch.tensor(costs_batch, dtype=torch.float, device=self.device)
        next_state_torch = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        next_action_torch = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)

        for head_idx in range(self.head_count):
            gamma = gamma_reward if head_idx == 0 else gamma_cost
            for member_idx in range(self.ensemble_size):
                net = self.head_members[head_idx][member_idx]
                target = self.target_head_members[head_idx][member_idx]
                optimizer = self.optimizers[head_idx][member_idx]
                loss_vec = self._td_loss(
                    net,
                    target,
                    func_value_torch,
                    head_idx,
                    state_torch,
                    action_torch,
                    costs_torch,
                    next_state_torch,
                    next_action_torch,
                )
                if self.ensemble_size > 1 and self.bootstrap_mask_prob < 1.0:
                    mask_rand = torch.rand(loss_vec.shape, generator=self.bootstrap_generator, dtype=loss_vec.dtype)
                    mask = (mask_rand.to(device=loss_vec.device) < self.bootstrap_mask_prob).float()
                    if torch.sum(mask) <= 0:
                        mask = torch.ones_like(loss_vec)
                    loss = torch.sum(loss_vec * mask) / torch.sum(mask)
                else:
                    loss = torch.mean(loss_vec)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                soft_update(target, net, gamma)

    def critic_value_stats(self, state_batch_torch, action_batch_torch):
        per_head_mean = []
        per_head_std = []
        for head_idx in range(self.head_count):
            member_values = []
            for member_idx in range(self.ensemble_size):
                target = self.target_head_members[head_idx][member_idx]
                values = target.forward(state_batch_torch, action_batch_torch).detach().view(-1)
                member_values.append(values)
            stacked = torch.stack(member_values, dim=1)
            per_head_mean.append(torch.mean(stacked, dim=1))
            per_head_std.append(torch.std(stacked, dim=1, unbiased=False))
        q_mean = torch.stack(per_head_mean, dim=1)
        q_std = torch.stack(per_head_std, dim=1)
        return q_mean, q_std

    def clone_as_legacy(self):
        legacy = copy.deepcopy(self)
        legacy.ensemble_size = 1
        legacy.head_members = [[members[0]] for members in legacy.head_members]
        legacy.target_head_members = [[members[0]] for members in legacy.target_head_members]
        legacy.optimizers = [[opts[0]] for opts in legacy.optimizers]
        return legacy
