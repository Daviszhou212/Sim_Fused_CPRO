import numpy as np
import torch.nn.functional as F
from model import Critic_net_MIMO
from model import Critic_net_CLQR
from utils import hard_update
from utils import soft_update
from utils import soft_update_twoloop
import torch



class Critic:

    def __init__(self, example_name, num_new_data, state_dim, action_dim, constraint_dim, q, device):
        self.example_name = example_name
        self.constraint_dim = constraint_dim
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.iter = 0
        self.num_new_data = num_new_data
        self.q=q

        # ----------------------
        if "MIMO" in self.example_name:
            # ----------------------
            self.net0 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net0 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic0_optimizer = torch.optim.Adam(self.net0.parameters(), 0.1/np.sqrt(self.q))
            hard_update(self.target_net0, self.net0)
            self.net1 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net1 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic1_optimizer = torch.optim.Adam(self.net1.parameters(), 0.1/np.sqrt(self.q))
            # ----------------------
            self.net2 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net2 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic2_optimizer = torch.optim.Adam(self.net2.parameters(), 0.1/np.sqrt(self.q))
            # ----------------------
            self.net3 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net3 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic3_optimizer = torch.optim.Adam(self.net3.parameters(), 0.1/np.sqrt(self.q))
            # ----------------------
            self.net4 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.target_net4 = Critic_net_MIMO(self.state_dim, self.action_dim, self.device)
            self.critic4_optimizer = torch.optim.Adam(self.net4.parameters(), 0.1/np.sqrt(self.q))
            hard_update(self.target_net1, self.net1)
            hard_update(self.target_net2, self.net2)
            hard_update(self.target_net3, self.net3)
            hard_update(self.target_net4, self.net4)
        else:
            # ----------------------
            self.net0 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.target_net0 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.critic0_optimizer = torch.optim.Adam(self.net0.parameters(), 0.1/np.sqrt(self.q))
            hard_update(self.target_net0, self.net0)
            self.net1 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.target_net1 = Critic_net_CLQR(self.state_dim, self.action_dim, self.device)
            self.critic1_optimizer = torch.optim.Adam(self.net1.parameters(), 0.005/np.sqrt(self.q))
            hard_update(self.target_net1, self.net1)



    def critic_update(self, func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost):
        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        costs_batch_torch = torch.tensor(costs_batch, dtype=torch.float, device=self.device)
        next_state_batch_torch = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        next_action_batch_torch = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        next_val0 = torch.squeeze(self.target_net0.forward(next_state_batch_torch, next_action_batch_torch).detach())
        y_expected0 = costs_batch_torch[:, 0] - func_value_torch[0] + next_val0
        y_predicted0 = torch.squeeze(self.net0.forward(state_batch_torch, action_batch_torch))
        loss_critic0 = F.smooth_l1_loss(y_predicted0, y_expected0)
        self.critic0_optimizer.zero_grad()
        loss_critic0.backward()
        self.critic0_optimizer.step()
        soft_update(self.target_net0, self.net0, gamma_reward)
        if "MIMO" in self.example_name:
            # ----------------------
            next_val1 = torch.squeeze(self.target_net1.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected1 = costs_batch_torch[:, 1] - func_value_torch[1] + next_val1
            y_predicted1 = torch.squeeze(self.net1.forward(state_batch_torch, action_batch_torch))
            loss_critic1 = F.smooth_l1_loss(y_predicted1, y_expected1)
            self.critic1_optimizer.zero_grad()
            loss_critic1.backward()
            self.critic1_optimizer.step()
            soft_update(self.target_net1, self.net1, gamma_cost)
            # ----------------------
            next_val2 = torch.squeeze(self.target_net2.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected2 = costs_batch_torch[:, 2] - func_value_torch[2] + next_val2
            y_predicted2 = torch.squeeze(self.net2.forward(state_batch_torch, action_batch_torch))
            loss_critic2 = F.smooth_l1_loss(y_predicted2, y_expected2)
            self.critic2_optimizer.zero_grad()
            loss_critic2.backward()
            self.critic2_optimizer.step()
            soft_update(self.target_net2, self.net2, gamma_cost)
            # ----------------------
            next_val3 = torch.squeeze(self.target_net3.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected3 = costs_batch_torch[:,3] - func_value_torch[3] + next_val3
            y_predicted3 = torch.squeeze(self.net3.forward(state_batch_torch, action_batch_torch))
            loss_critic3 = F.smooth_l1_loss(y_predicted3, y_expected3)
            self.critic3_optimizer.zero_grad()
            loss_critic3.backward()
            self.critic3_optimizer.step()
            soft_update(self.target_net3, self.net3, gamma_cost)
            # ----------------------
            next_val4 = torch.squeeze(self.target_net4.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected4 = costs_batch_torch[:, 4] - func_value_torch[4] + next_val4
            y_predicted4 = torch.squeeze(self.net4.forward(state_batch_torch, action_batch_torch))
            loss_critic4 = F.smooth_l1_loss(y_predicted4, y_expected4)
            self.critic4_optimizer.zero_grad()
            loss_critic4.backward()
            self.critic4_optimizer.step()
            soft_update(self.target_net4, self.net4, gamma_cost)
        else:
            # ----------------------
            next_val1 = torch.squeeze(self.target_net1.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected1 = costs_batch_torch[:, 1] - func_value_torch[1] + next_val1
            y_predicted1 = torch.squeeze(self.net1.forward(state_batch_torch, action_batch_torch))
            loss_critic1 = F.smooth_l1_loss(y_predicted1, y_expected1)
            self.critic1_optimizer.zero_grad()
            loss_critic1.backward()
            self.critic1_optimizer.step()
            soft_update(self.target_net1, self.net1, gamma_cost)

    def critic_update_twoloop(self, func_value, state_batch, action_batch, costs_batch, next_state_batch, next_action_batch, eta, gamma_reward, gamma_cost, Q_update_time, Q_update_index):
        func_value_torch = torch.tensor(func_value, dtype=torch.float, device=self.device)
        state_batch_torch = torch.tensor(state_batch, dtype=torch.float, device=self.device)
        action_batch_torch = torch.tensor(action_batch, dtype=torch.float, device=self.device)
        costs_batch_torch = torch.tensor(costs_batch, dtype=torch.float, device=self.device)
        next_state_batch_torch = torch.tensor(next_state_batch, dtype=torch.float, device=self.device)
        next_action_batch_torch = torch.tensor(next_action_batch, dtype=torch.float, device=self.device)
        # ---------------------- optimize critic ----------------------
        # Use target actor exploitation policy here for loss evaluation
        if "MIMO" in self.example_name:
            # ----------------------
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
            # ----------------------
            next_val2 = torch.squeeze(self.target_net2.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected2 = costs_batch_torch[:, 2] - func_value_torch[2] + next_val2
            y_predicted2 = torch.squeeze(self.net2.forward(state_batch_torch, action_batch_torch))
            loss_critic2 = F.smooth_l1_loss(y_predicted2, y_expected2)
            self.critic2_optimizer.zero_grad()
            loss_critic2.backward()
            self.critic2_optimizer.step()
            # ----------------------
            next_val3 = torch.squeeze(self.target_net3.forward(next_state_batch_torch, next_action_batch_torch).detach())
            y_expected3 = costs_batch_torch[:,3] - func_value_torch[3] + next_val3
            y_predicted3 = torch.squeeze(self.net3.forward(state_batch_torch, action_batch_torch))
            loss_critic3 = F.smooth_l1_loss(y_predicted3, y_expected3)
            self.critic3_optimizer.zero_grad()
            loss_critic3.backward()
            self.critic3_optimizer.step()
            # ----------------------
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
            # ----------------------
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

    def critic_value(self, state_batch_torch, action_batch_torch):
        Q_hat = np.matrix(np.zeros((self.num_new_data, 1 + self.constraint_dim)))
        Q_hat[:, 0] = self.target_net0.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        if "MIMO" in self.example_name:
            Q_hat[:, 1] = self.target_net1.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
            Q_hat[:, 2] = self.target_net2.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
            Q_hat[:, 3] = self.target_net3.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
            Q_hat[:, 4] = self.target_net4.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        else:
            Q_hat[:, 1] = self.net1.forward(state_batch_torch, action_batch_torch).detach().cpu().numpy()
        Q_hat_torch = torch.tensor(Q_hat, dtype=torch.float, device=self.device)

        return Q_hat_torch


