import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_q_net(input_dim, hidden_dims):
    layers = []
    prev_dim = int(input_dim)
    for hidden_dim in tuple(hidden_dims):
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.Tanh())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, 1))
    return nn.Sequential(*layers)


class MultiHeadDifferentialCritic:
    def __init__(self, state_dim, action_dim, constraint_dim, hidden_dims=(64, 64), device="cpu", learning_rate=1e-3):
        self.state_dim = int(state_dim)
        self.action_dim = int(action_dim)
        self.constraint_dim = int(constraint_dim)
        self.cost_dim = 1 + self.constraint_dim
        self.device = torch.device(device)
        self.input_dim = self.state_dim + self.action_dim
        self.heads = nn.ModuleList([_build_q_net(self.input_dim, hidden_dims) for _ in range(self.cost_dim)]).to(self.device)
        self.target_heads = nn.ModuleList([_build_q_net(self.input_dim, hidden_dims) for _ in range(self.cost_dim)]).to(self.device)
        for target, source in zip(self.target_heads, self.heads):
            target.load_state_dict(source.state_dict())
        self.optimizers = [torch.optim.Adam(head.parameters(), lr=float(learning_rate)) for head in self.heads]

    def _input(self, state, action):
        state = torch.as_tensor(state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        return torch.cat((state, action), dim=-1)

    def _values(self, modules, state, action):
        x = self._input(state, action)
        return torch.stack([module(x).reshape(-1) for module in modules], dim=1)

    def online_value(self, state, action):
        return self._values(self.heads, state, action)

    def target_value(self, state, action):
        return self._values(self.target_heads, state, action)

    def compute_td_target(self, costs, next_state, next_action, func_value, critic_target_mode):
        costs = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
        func_value = torch.as_tensor(func_value, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            if critic_target_mode == "source_compatible":
                next_value = self.target_value(next_state, next_action)
            elif critic_target_mode == "tex_strict":
                next_value = self.online_value(next_state, next_action)
            else:
                raise ValueError("unsupported critic_target_mode: {0}".format(critic_target_mode))
            return costs - func_value + next_value

    def _soft_update(self, gamma):
        gamma = float(gamma)
        with torch.no_grad():
            for target, source in zip(self.target_heads, self.heads):
                for target_param, source_param in zip(target.parameters(), source.parameters()):
                    target_param.mul_(1.0 - gamma).add_(source_param, alpha=gamma)

    def update(self, state, action, costs, next_state, next_action, func_value, eta, gamma, critic_target_mode):
        target = self.compute_td_target(costs, next_state, next_action, func_value, critic_target_mode)
        x = self._input(state, action)
        losses = []
        for head_idx, optimizer in enumerate(self.optimizers):
            for group in optimizer.param_groups:
                group["lr"] = float(eta)
            current = self.heads[head_idx](x).reshape(-1)
            loss = F.smooth_l1_loss(current, target[:, head_idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach().cpu().item()))
        self._soft_update(float(gamma))
        return losses

    def critic_value(self, state, action, use_target=True):
        with torch.no_grad():
            return self.target_value(state, action) if use_target else self.online_value(state, action)

    def checkpoint_state(self):
        state = {}
        for idx, module in enumerate(self.heads):
            for key, value in module.state_dict().items():
                state["heads.{0}.{1}".format(idx, key)] = value.detach().cpu().clone()
        for idx, module in enumerate(self.target_heads):
            for key, value in module.state_dict().items():
                state["target_heads.{0}.{1}".format(idx, key)] = value.detach().cpu().clone()
        return state
