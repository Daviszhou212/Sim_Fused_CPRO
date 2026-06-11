import numpy as np
import torch
import torch.nn.functional as F

from model import Critic_net_CLQR
from model import Critic_net_MIMO
from utils import hard_update
from utils import soft_update
from utils import soft_update_twoloop


def _flatten_network_params(modules):
    params = []
    for module in modules:
        for para in module.parameters():
            params.append(para.detach().view(-1).cpu())
    if not params:
        return np.zeros((0,), dtype=np.float64)
    return torch.cat(params).numpy().astype(np.float64, copy=False)


class Critic:
    def __init__(self, example_name, num_new_data, state_dim, action_dim, constraint_dim, q, device):
        self.example_name = example_name
        self.constraint_dim = constraint_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.iter = 0
        self.num_new_data = num_new_data
        self.q = q
        self.head_nets = []
        self.target_head_nets = []
        self.head_optimizers = []
        self.head_base_lrs = []

        if "MIMO" in self.example_name:
            for head_idx in range(1 + int(self.constraint_dim)):
                self._create_head(head_idx, Critic_net_MIMO, 0.1 / np.sqrt(self.q))
        else:
            self._create_head(0, Critic_net_CLQR, 0.1 / np.sqrt(self.q))
            self._create_head(1, Critic_net_CLQR, 0.005 / np.sqrt(self.q))

    def _create_head(self, head_idx, net_cls, base_lr):
        net = net_cls(self.state_dim, self.action_dim, self.device)
        target_net = net_cls(self.state_dim, self.action_dim, self.device)
        optimizer = torch.optim.Adam(net.parameters(), float(base_lr))
        hard_update(target_net, net)
        self.head_nets.append(net)
        self.target_head_nets.append(target_net)
        self.head_optimizers.append(optimizer)
        self.head_base_lrs.append(float(base_lr))
        setattr(self, "net{0}".format(head_idx), net)
        setattr(self, "target_net{0}".format(head_idx), target_net)
        setattr(self, "critic{0}_base_lr".format(head_idx), float(base_lr))
        setattr(self, "critic{0}_optimizer".format(head_idx), optimizer)

    def _set_optimizer_lr(self, optimizer, lr_value):
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr_value)

    def _build_head_specs(self, gamma_reward, gamma_cost):
        specs = []
        for head_idx, (net, target_net, optimizer, base_lr) in enumerate(
            zip(self.head_nets, self.target_head_nets, self.head_optimizers, self.head_base_lrs)
        ):
            gamma = gamma_reward if head_idx == 0 else gamma_cost
            specs.append((net, target_net, optimizer, base_lr, gamma))
        return specs

    def _compute_td_loss(
        self,
        net,
        target_net,
        func_value_torch,
        head_idx,
        state_batch_torch,
        action_batch_torch,
        costs_batch_torch,
        next_state_batch_torch,
        next_action_batch_torch,
    ):
        next_val = torch.squeeze(target_net.forward(next_state_batch_torch, next_action_batch_torch).detach())
        y_expected = costs_batch_torch[:, head_idx] - func_value_torch[head_idx] + next_val
        y_predicted = torch.squeeze(net.forward(state_batch_torch, action_batch_torch))
        return F.smooth_l1_loss(y_predicted, y_expected)

    def _compute_mixed_td_loss(
        self,
        net,
        target_net,
        func_value_torch,
        head_idx,
        online_state_batch_torch,
        online_action_batch_torch,
        online_costs_batch_torch,
        online_next_state_batch_torch,
        online_next_action_batch_torch,
        offline_state_batch_torch,
        offline_action_batch_torch,
        offline_costs_batch_torch,
        offline_next_state_batch_torch,
        offline_next_action_batch_torch,
        critic_xi,
    ):
        loss_online = self._compute_td_loss(
            net,
            target_net,
            func_value_torch,
            head_idx,
            online_state_batch_torch,
            online_action_batch_torch,
            online_costs_batch_torch,
            online_next_state_batch_torch,
            online_next_action_batch_torch,
        )
        if offline_state_batch_torch is None:
            return loss_online

        xi = float(np.clip(float(critic_xi), 0.0, 1.0))
        if xi <= 0.0:
            return loss_online

        loss_offline = self._compute_td_loss(
            net,
            target_net,
            func_value_torch,
            head_idx,
            offline_state_batch_torch,
            offline_action_batch_torch,
            offline_costs_batch_torch,
            offline_next_state_batch_torch,
            offline_next_action_batch_torch,
        )
        return (1.0 - xi) * loss_online + xi * loss_offline

    def _legacy_single_head_update(
        self,
        net,
        target_net,
        optimizer,
        base_lr,
        func_value_torch,
        head_idx,
        state_batch_torch,
        action_batch_torch,
        costs_batch_torch,
        next_state_batch_torch,
        next_action_batch_torch,
        gamma,
    ):
        self._set_optimizer_lr(optimizer, base_lr)
        next_val = torch.squeeze(target_net.forward(next_state_batch_torch, next_action_batch_torch).detach())
        y_expected = costs_batch_torch[:, head_idx] - func_value_torch[head_idx] + next_val
        y_predicted = torch.squeeze(net.forward(state_batch_torch, action_batch_torch))
        loss = F.smooth_l1_loss(y_predicted, y_expected)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        soft_update(target_net, net, gamma)

    def _mixed_single_head_update(
        self,
        net,
        target_net,
        optimizer,
        base_lr,
        func_value_torch,
        head_idx,
        online_state_batch_torch,
        online_action_batch_torch,
        online_costs_batch_torch,
        online_next_state_batch_torch,
        online_next_action_batch_torch,
        offline_state_batch_torch,
        offline_action_batch_torch,
        offline_costs_batch_torch,
        offline_next_state_batch_torch,
        offline_next_action_batch_torch,
        critic_xi,
        gamma,
    ):
        self._set_optimizer_lr(optimizer, base_lr)
        loss = self._compute_mixed_td_loss(
            net,
            target_net,
            func_value_torch,
            head_idx,
            online_state_batch_torch,
            online_action_batch_torch,
            online_costs_batch_torch,
            online_next_state_batch_torch,
            online_next_action_batch_torch,
            offline_state_batch_torch,
            offline_action_batch_torch,
            offline_costs_batch_torch,
            offline_next_state_batch_torch,
            offline_next_action_batch_torch,
            critic_xi,
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        soft_update(target_net, net, gamma)

    def critic_update(
        self,
        func_value,
        state_batch,
        action_batch,
        costs_batch,
        next_state_batch,
        next_action_batch,
        eta,
        gamma_reward,
        gamma_cost,
        offline_state_batch=None,
        offline_action_batch=None,
        offline_costs_batch=None,
        offline_next_state_batch=None,
        offline_next_action_batch=None,
        critic_xi=0.0,
    ):
        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        online_state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        online_action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        online_costs_batch_torch = torch.tensor(costs_batch, dtype=torch.float, device=self.device)
        online_next_state_batch_torch = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        online_next_action_batch_torch = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)

        use_offline_batch = (
            offline_state_batch is not None
            and offline_action_batch is not None
            and offline_costs_batch is not None
            and offline_next_state_batch is not None
            and offline_next_action_batch is not None
            and np.asarray(offline_state_batch).shape[0] > 0
        )
        if use_offline_batch:
            offline_state_batch_torch = torch.tensor(offline_state_batch, dtype=torch.float, device=self.device)
            offline_action_batch_torch = torch.tensor(offline_action_batch, dtype=torch.float, device=self.device)
            offline_costs_batch_torch = torch.tensor(offline_costs_batch, dtype=torch.float, device=self.device)
            offline_next_state_batch_torch = torch.tensor(offline_next_state_batch, dtype=torch.float, device=self.device)
            offline_next_action_batch_torch = torch.tensor(offline_next_action_batch, dtype=torch.float, device=self.device)
        else:
            offline_state_batch_torch = None
            offline_action_batch_torch = None
            offline_costs_batch_torch = None
            offline_next_state_batch_torch = None
            offline_next_action_batch_torch = None

        use_legacy_online_mode = (not use_offline_batch) or (float(critic_xi) <= 0.0)
        for head_idx, (net, target_net, optimizer, base_lr, gamma) in enumerate(
            self._build_head_specs(gamma_reward, gamma_cost)
        ):
            if use_legacy_online_mode:
                self._legacy_single_head_update(
                    net,
                    target_net,
                    optimizer,
                    base_lr,
                    func_value_torch,
                    head_idx,
                    online_state_batch_torch,
                    online_action_batch_torch,
                    online_costs_batch_torch,
                    online_next_state_batch_torch,
                    online_next_action_batch_torch,
                    gamma,
                )
            else:
                self._mixed_single_head_update(
                    net,
                    target_net,
                    optimizer,
                    base_lr,
                    func_value_torch,
                    head_idx,
                    online_state_batch_torch,
                    online_action_batch_torch,
                    online_costs_batch_torch,
                    online_next_state_batch_torch,
                    online_next_action_batch_torch,
                    offline_state_batch_torch,
                    offline_action_batch_torch,
                    offline_costs_batch_torch,
                    offline_next_state_batch_torch,
                    offline_next_action_batch_torch,
                    critic_xi,
                    gamma,
                )

    def critic_update_twoloop(self, func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost, Q_update_time, Q_update_index):
        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        costs_batch_torch = torch.tensor(costs_batch, dtype=torch.float, device=self.device)
        next_state_batch_torch = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        next_action_batch_torch = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)
        for head_idx, (net, target_net, optimizer, _, gamma) in enumerate(
            self._build_head_specs(gamma_reward, gamma_cost)
        ):
            next_val = torch.squeeze(target_net.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected = costs_batch_torch[:, head_idx] - func_value_torch[head_idx] + next_val
            y_predicted = torch.squeeze(net.forward(state_batch_torch, action_batch_torch))
            loss_critic = F.smooth_l1_loss(y_predicted, y_expected)
            optimizer.zero_grad()
            loss_critic.backward()
            optimizer.step()
            soft_update_twoloop(target_net, net, gamma, Q_update_time, Q_update_index)

    def critic_value(self, state_batch_torch, action_batch_torch, legacy_online_mode=False):
        batch_size = int(state_batch_torch.shape[0])
        q_hat = np.zeros((batch_size, 1 + self.constraint_dim), dtype=np.float64)
        for head_idx in range(1 + int(self.constraint_dim)):
            if (head_idx == 1) and ("MIMO" not in self.example_name) and bool(legacy_online_mode):
                net = self.head_nets[head_idx]
            else:
                net = self.target_head_nets[head_idx]
            q_hat[:, head_idx] = net.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
        return torch.tensor(q_hat, dtype=torch.float, device=self.device)

    def flatten_parameters(self, include_target=True):
        modules = list(self.head_nets)
        if include_target:
            modules.extend(self.target_head_nets)
        return _flatten_network_params(modules)
