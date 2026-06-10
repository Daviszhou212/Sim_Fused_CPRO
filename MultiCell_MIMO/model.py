import numpy as np
import torch
import torch.nn as nn


ACTION_EPS = 1e-6


def _build_mlp(input_dim, hidden_dims, output_dim):
    layers = []
    prev_dim = int(input_dim)
    for hidden_dim in tuple(hidden_dims):
        layer = nn.Linear(prev_dim, int(hidden_dim))
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2.0))
        nn.init.constant_(layer.bias, 0.0)
        layers.append(layer)
        layers.append(nn.Tanh())
        prev_dim = int(hidden_dim)
    output = nn.Linear(prev_dim, int(output_dim))
    nn.init.orthogonal_(output.weight, gain=np.sqrt(2.0))
    nn.init.constant_(output.bias, 0.0)
    layers.append(output)
    return nn.Sequential(*layers)


class SharedLocalGaussianActor(nn.Module):
    def __init__(
        self,
        local_state_dim,
        users_per_cell,
        cell_count,
        hidden_dims=(128, 128),
        device="cpu",
        power_max=2.5,
    ):
        super().__init__()
        self.local_state_dim = int(local_state_dim)
        self.users_per_cell = int(users_per_cell)
        self.cell_count = int(cell_count)
        self.cell_action_dim = self.users_per_cell + 1
        self.action_dim = self.cell_count * self.cell_action_dim
        self.power_max = float(power_max)
        self.device = torch.device(device)
        self.net = _build_mlp(self.local_state_dim, tuple(hidden_dims), self.cell_action_dim)
        self.log_std = nn.Parameter(-0.5 * torch.ones(self.cell_action_dim, dtype=torch.float32))
        self.to(self.device)

    def _as_local_batch(self, local_states):
        tensor = torch.as_tensor(local_states, dtype=torch.float32, device=self.device)
        if tensor.dim() == 2:
            if tensor.shape[0] != self.cell_count:
                raise ValueError("local state cell dimension mismatch")
            tensor = tensor.unsqueeze(0)
        if tensor.dim() != 3:
            raise ValueError("local_states must have shape (cell, dim) or (batch, cell, dim)")
        if tensor.shape[1] != self.cell_count or tensor.shape[2] != self.local_state_dim:
            raise ValueError("local_states shape mismatch")
        return tensor

    def _local_raw_mean(self, local_states_2d):
        return self.power_max * torch.sigmoid(self.net(local_states_2d))

    def _raw_mean_batch(self, local_states):
        local_states = self._as_local_batch(local_states)
        batch_size = local_states.shape[0]
        flat_state = local_states.reshape(batch_size * self.cell_count, self.local_state_dim)
        raw = self._local_raw_mean(flat_state)
        return raw.reshape(batch_size, self.cell_count, self.cell_action_dim)

    def sample_action_tensor(self, local_states, use_mean=False, reparameterized=False):
        mean = self._raw_mean_batch(local_states)
        if use_mean:
            action = mean
        else:
            std = torch.exp(self.log_std).view(1, 1, self.cell_action_dim)
            dist = torch.distributions.Normal(mean, std)
            action = dist.rsample() if reparameterized else dist.sample()
        return action.reshape(action.shape[0], self.action_dim)

    def sample_action(self, local_states, use_mean=False):
        self.eval()
        with torch.no_grad():
            action = self.sample_action_tensor(local_states, use_mean=use_mean)
        return action.reshape(-1).detach().cpu().numpy()

    def sample_cell_action(self, local_state, use_mean=True):
        local_state = torch.as_tensor(local_state, dtype=torch.float32, device=self.device).view(1, -1)
        with torch.no_grad():
            mean = self._local_raw_mean(local_state)
            action = mean if use_mean else torch.distributions.Normal(mean, torch.exp(self.log_std)).sample()
        return action.reshape(-1).detach().cpu().numpy()

    def evaluate_cells(self, local_states, action):
        local_states = self._as_local_batch(local_states)
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.dim() == 1:
            action = action.view(1, self.action_dim)
        action = action.view(local_states.shape[0], self.cell_count, self.cell_action_dim)
        mean = self._raw_mean_batch(local_states)
        std = torch.exp(self.log_std).view(1, 1, self.cell_action_dim)
        dist = torch.distributions.Normal(mean, std)
        return dist.log_prob(action).sum(dim=-1)

    def evaluate_action(self, local_states, action):
        return self.evaluate_cells(local_states, action).sum(dim=1)

    def flatten_parameters(self):
        parts = [param.detach().reshape(-1) for param in self.net.parameters()]
        parts.append(self.log_std.detach().reshape(-1))
        return torch.cat(parts)

    def flatten_grad(self):
        parts = []
        for param in self.net.parameters():
            if param.grad is None:
                parts.append(torch.zeros_like(param).reshape(-1))
            else:
                parts.append(param.grad.reshape(-1))
        if self.log_std.grad is None:
            parts.append(torch.zeros_like(self.log_std).reshape(-1))
        else:
            parts.append(self.log_std.grad.reshape(-1))
        return torch.cat(parts)

    def restore_parameters(self, flat_parameters):
        flat = torch.as_tensor(flat_parameters, dtype=torch.float32, device=self.device).reshape(-1)
        offset = 0
        with torch.no_grad():
            for param in self.net.parameters():
                count = param.numel()
                param.copy_(flat[offset : offset + count].view_as(param))
                offset += count
            count = self.log_std.numel()
            self.log_std.copy_(flat[offset : offset + count].view_as(self.log_std))
            offset += count
        if offset != flat.numel():
            raise ValueError("unused actor parameter entries: {0}".format(flat.numel() - offset))

    def checkpoint_state(self):
        state = {f"net.{key}": value.detach().cpu().clone() for key, value in self.net.state_dict().items()}
        state["log_std"] = self.log_std.detach().cpu().clone()
        return state
