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

        if "MIMO" in self.example_name:
            self.net0 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net0 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic0_base_lr = 0.1 / np.sqrt(self.q)
            self.critic0_optimizer = torch.optim.Adam(self.net0.parameters(), self.critic0_base_lr)
            hard_update(self.target_net0, self.net0)

            self.net1 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net1 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic1_base_lr = 0.1 / np.sqrt(self.q)
            self.critic1_optimizer = torch.optim.Adam(self.net1.parameters(), self.critic1_base_lr)

            self.net2 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net2 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic2_base_lr = 0.1 / np.sqrt(self.q)
            self.critic2_optimizer = torch.optim.Adam(self.net2.parameters(), self.critic2_base_lr)

            self.net3 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net3 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic3_base_lr = 0.1 / np.sqrt(self.q)
            self.critic3_optimizer = torch.optim.Adam(self.net3.parameters(), self.critic3_base_lr)

            self.net4 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net4 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic4_base_lr = 0.1 / np.sqrt(self.q)
            self.critic4_optimizer = torch.optim.Adam(self.net4.parameters(), self.critic4_base_lr)

            hard_update(self.target_net1, self.net1)
            hard_update(self.target_net2, self.net2)
            hard_update(self.target_net3, self.net3)
            hard_update(self.target_net4, self.net4)
        else:
            self.net0 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.target_net0 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.critic0_base_lr = 0.1 / np.sqrt(self.q)
            self.critic0_optimizer = torch.optim.Adam(self.net0.parameters(), self.critic0_base_lr)
            hard_update(self.target_net0, self.net0)

            self.net1 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.target_net1 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.critic1_base_lr = 0.005 / np.sqrt(self.q)
            self.critic1_optimizer = torch.optim.Adam(self.net1.parameters(), self.critic1_base_lr)
            hard_update(self.target_net1, self.net1)

    def _set_optimizer_lr(self, optimizer, lr_value):
        for param_group in optimizer.param_groups:
            param_group["lr"] = float(lr_value)

    def _build_head_specs(self, gamma_reward, gamma_cost):
        specs = [
            (self.net0, self.target_net0, self.critic0_optimizer, self.critic0_base_lr, gamma_reward),
            (self.net1, self.target_net1, self.critic1_optimizer, self.critic1_base_lr, gamma_cost),
        ]
        if "MIMO" in self.example_name:
            specs.extend(
                [
                    (self.net2, self.target_net2, self.critic2_optimizer, self.critic2_base_lr, gamma_cost),
                    (self.net3, self.target_net3, self.critic3_optimizer, self.critic3_base_lr, gamma_cost),
                    (self.net4, self.target_net4, self.critic4_optimizer, self.critic4_base_lr, gamma_cost),
                ]
            )
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
        if "MIMO" in self.example_name:
            next_val0 = torch.squeeze(self.target_net0.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected0 = costs_batch_torch[:, 0] - func_value_torch[0] + next_val0
            y_predicted0 = torch.squeeze(self.net0.forward(state_batch_torch, action_batch_torch))
            loss_critic0 = F.smooth_l1_loss(y_predicted0, y_expected0)
            self.critic0_optimizer.zero_grad()
            loss_critic0.backward()
            self.critic0_optimizer.step()

            next_val1 = torch.squeeze(self.target_net1.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected1 = costs_batch_torch[:, 1] - func_value_torch[1] + next_val1
            y_predicted1 = torch.squeeze(self.net1.forward(state_batch_torch, action_batch_torch))
            loss_critic1 = F.smooth_l1_loss(y_predicted1, y_expected1)
            self.critic1_optimizer.zero_grad()
            loss_critic1.backward()
            self.critic1_optimizer.step()

            next_val2 = torch.squeeze(self.target_net2.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected2 = costs_batch_torch[:, 2] - func_value_torch[2] + next_val2
            y_predicted2 = torch.squeeze(self.net2.forward(state_batch_torch, action_batch_torch))
            loss_critic2 = F.smooth_l1_loss(y_predicted2, y_expected2)
            self.critic2_optimizer.zero_grad()
            loss_critic2.backward()
            self.critic2_optimizer.step()

            next_val3 = torch.squeeze(self.target_net3.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected3 = costs_batch_torch[:, 3] - func_value_torch[3] + next_val3
            y_predicted3 = torch.squeeze(self.net3.forward(state_batch_torch, action_batch_torch))
            loss_critic3 = F.smooth_l1_loss(y_predicted3, y_expected3)
            self.critic3_optimizer.zero_grad()
            loss_critic3.backward()
            self.critic3_optimizer.step()

            next_val4 = torch.squeeze(self.target_net4.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected4 = costs_batch_torch[:, 4] - func_value_torch[4] + next_val4
            y_predicted4 = torch.squeeze(self.net4.forward(state_batch_torch, action_batch_torch))
            loss_critic4 = F.smooth_l1_loss(y_predicted4, y_expected4)
            self.critic4_optimizer.zero_grad()
            loss_critic4.backward()
            self.critic4_optimizer.step()

            soft_update_twoloop(self.target_net0, self.net0, gamma_reward, Q_update_time, Q_update_index)
            soft_update_twoloop(self.target_net1, self.net1, gamma_cost, Q_update_time, Q_update_index)
            soft_update_twoloop(self.target_net2, self.net2, gamma_cost, Q_update_time, Q_update_index)
            soft_update_twoloop(self.target_net3, self.net3, gamma_cost, Q_update_time, Q_update_index)
            soft_update_twoloop(self.target_net4, self.net4, gamma_cost, Q_update_time, Q_update_index)
        else:
            next_val0 = torch.squeeze(self.target_net0.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected0 = costs_batch_torch[:, 0] - func_value_torch[0] + next_val0
            y_predicted0 = torch.squeeze(self.net0.forward(state_batch_torch, action_batch_torch))
            loss_critic0 = F.smooth_l1_loss(y_predicted0, y_expected0)
            self.critic0_optimizer.zero_grad()
            loss_critic0.backward()
            self.critic0_optimizer.step()

            next_val1 = torch.squeeze(self.target_net1.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected1 = costs_batch_torch[:, 1] - func_value_torch[1] + next_val1
            y_predicted1 = torch.squeeze(self.net1.forward(state_batch_torch, action_batch_torch))
            loss_critic1 = F.smooth_l1_loss(y_predicted1, y_expected1)
            self.critic1_optimizer.zero_grad()
            loss_critic1.backward()
            self.critic1_optimizer.step()

            soft_update_twoloop(self.target_net0, self.net0, gamma_reward, Q_update_time, Q_update_index)
            soft_update_twoloop(self.target_net1, self.net1, gamma_cost, Q_update_time, Q_update_index)

    def critic_value(self, state_batch_torch, action_batch_torch, legacy_online_mode=False):
        batch_size = int(state_batch_torch.shape[0])
        q_hat = np.zeros((batch_size, 1 + self.constraint_dim), dtype=np.float64)
        q_hat[:, 0] = self.target_net0.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
        if "MIMO" in self.example_name:
            q_hat[:, 1] = self.target_net1.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
            q_hat[:, 2] = self.target_net2.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
            q_hat[:, 3] = self.target_net3.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
            q_hat[:, 4] = self.target_net4.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
        else:
            constraint_net = self.net1 if bool(legacy_online_mode) else self.target_net1
            q_hat[:, 1] = constraint_net.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy().reshape(-1)
        return torch.tensor(q_hat, dtype=torch.float, device=self.device)

    def flatten_parameters(self, include_target=True):
        modules = [self.net0, self.net1]
        if "MIMO" in self.example_name:
            modules.extend([self.net2, self.net3, self.net4])
        if include_target:
            modules.extend([self.target_net0, self.target_net1])
            if "MIMO" in self.example_name:
                modules.extend([self.target_net2, self.target_net3, self.target_net4])
        return _flatten_network_params(modules)
