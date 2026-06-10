import torch
import torch.nn as nn
import torch.nn.functional as F


def _build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = int(input_dim)
    for hidden_dim in tuple(hidden_dims):
        layers.append(nn.Linear(prev_dim, int(hidden_dim)))
        layers.append(nn.Tanh())
        prev_dim = int(hidden_dim)
    layers.append(nn.Linear(prev_dim, int(output_dim)))
    return nn.Sequential(*layers)


class TreeMessageDifferentialCritic:
    """Tree-bottleneck critic for the CMARL v2 design.

    The target modules are the averaged critic copy: bar_phi plus bar_omega.
    They are not discounted critics; updates are pure parameter averaging.
    """

    def __init__(
        self,
        local_state_dim,
        cell_count,
        cell_action_dim,
        constraint_dim,
        message_dim=32,
        hidden_dims=(64, 64),
        device="cpu",
        learning_rate=1e-3,
    ):
        self.local_state_dim = int(local_state_dim)
        self.cell_count = int(cell_count)
        self.cell_action_dim = int(cell_action_dim)
        self.constraint_dim = int(constraint_dim)
        self.cost_dim = 1 + self.constraint_dim
        self.message_dim = int(message_dim)
        self.device = torch.device(device)
        encoder_input_dim = self.local_state_dim + self.cell_action_dim
        self.encoder = _build_mlp(encoder_input_dim, tuple(hidden_dims), self.message_dim).to(self.device)
        self.target_encoder = _build_mlp(encoder_input_dim, tuple(hidden_dims), self.message_dim).to(self.device)
        self.heads = nn.ModuleList([_build_mlp(self.message_dim, (), 1) for _ in range(self.cost_dim)]).to(self.device)
        self.target_heads = nn.ModuleList([_build_mlp(self.message_dim, (), 1) for _ in range(self.cost_dim)]).to(self.device)
        self.target_encoder.load_state_dict(self.encoder.state_dict())
        for target, source in zip(self.target_heads, self.heads):
            target.load_state_dict(source.state_dict())
        params = list(self.encoder.parameters()) + list(self.heads.parameters())
        self.optimizer = torch.optim.Adam(params, lr=float(learning_rate))

    def _as_batches(self, local_state, action):
        local_state = torch.as_tensor(local_state, dtype=torch.float32, device=self.device)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if local_state.dim() != 3:
            raise ValueError("local_state must have shape (batch, cell, local_state_dim)")
        if local_state.shape[1] != self.cell_count or local_state.shape[2] != self.local_state_dim:
            raise ValueError("local_state shape mismatch")
        if action.dim() == 1:
            action = action.view(1, -1)
        if action.shape[-1] != self.cell_count * self.cell_action_dim:
            raise ValueError("action dimension mismatch")
        cell_action = action.view(local_state.shape[0], self.cell_count, self.cell_action_dim)
        return local_state, cell_action

    def _tree_reduce(self, messages):
        current = messages
        while current.shape[1] > 1:
            pair_count = current.shape[1] // 2
            paired = current[:, : pair_count * 2, :].reshape(current.shape[0], pair_count, 2, self.message_dim)
            reduced = paired.sum(dim=2)
            if current.shape[1] % 2 == 1:
                reduced = torch.cat((reduced, current[:, -1:, :]), dim=1)
            current = reduced
        return current[:, 0, :] / float(max(self.cell_count, 1))

    def _encode(self, encoder, local_state, action):
        local_state, cell_action = self._as_batches(local_state, action)
        local_input = torch.cat((local_state, cell_action), dim=-1)
        flat_input = local_input.reshape(local_state.shape[0] * self.cell_count, -1)
        messages = encoder(flat_input).reshape(local_state.shape[0], self.cell_count, self.message_dim)
        return self._tree_reduce(messages)

    def _values(self, encoder, heads, local_state, action):
        z = self._encode(encoder, local_state, action)
        return torch.stack([head(z).reshape(-1) for head in heads], dim=1)

    def online_value(self, local_state, action):
        return self._values(self.encoder, self.heads, local_state, action)

    def target_value(self, local_state, action):
        return self._values(self.target_encoder, self.target_heads, local_state, action)

    def compute_td_target(self, costs, next_local_state, next_action, func_value, critic_target_mode):
        costs = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
        func_value = torch.as_tensor(func_value, dtype=torch.float32, device=self.device).reshape(1, -1)
        with torch.no_grad():
            if critic_target_mode == "source_compatible":
                next_value = self.target_value(next_local_state, next_action)
            elif critic_target_mode == "tex_strict":
                next_value = self.online_value(next_local_state, next_action)
            else:
                raise ValueError("unsupported critic_target_mode: {0}".format(critic_target_mode))
            return costs - func_value + next_value

    def _soft_update(self, gamma):
        gamma = float(gamma)
        with torch.no_grad():
            for target_param, source_param in zip(self.target_encoder.parameters(), self.encoder.parameters()):
                target_param.mul_(1.0 - gamma).add_(source_param, alpha=gamma)
            for target, source in zip(self.target_heads, self.heads):
                for target_param, source_param in zip(target.parameters(), source.parameters()):
                    target_param.mul_(1.0 - gamma).add_(source_param, alpha=gamma)

    def update(self, local_state, action, costs, next_local_state, next_action, func_value, eta, gamma, critic_target_mode):
        target = self.compute_td_target(costs, next_local_state, next_action, func_value, critic_target_mode)
        current = self.online_value(local_state, action)
        losses = [F.smooth_l1_loss(current[:, idx], target[:, idx]) for idx in range(self.cost_dim)]
        total_loss = sum(losses)
        for group in self.optimizer.param_groups:
            group["lr"] = float(eta)
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        self._soft_update(gamma)
        return [float(loss.detach().cpu().item()) for loss in losses]

    def flatten_target_parameters(self):
        parts = [param.detach().reshape(-1) for param in self.target_encoder.parameters()]
        for head in self.target_heads:
            parts.extend(param.detach().reshape(-1) for param in head.parameters())
        return torch.cat(parts)

    def checkpoint_state(self):
        state = {}
        for key, value in self.encoder.state_dict().items():
            state["encoder.{0}".format(key)] = value.detach().cpu().clone()
        for key, value in self.target_encoder.state_dict().items():
            state["target_encoder.{0}".format(key)] = value.detach().cpu().clone()
        for idx, head in enumerate(self.heads):
            for key, value in head.state_dict().items():
                state["heads.{0}.{1}".format(idx, key)] = value.detach().cpu().clone()
        for idx, head in enumerate(self.target_heads):
            for key, value in head.state_dict().items():
                state["target_heads.{0}.{1}".format(idx, key)] = value.detach().cpu().clone()
        return state
