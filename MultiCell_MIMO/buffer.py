import numpy as np


class TransitionBuffer:
    def __init__(self, capacity, state_dim, action_dim, cost_dim):
        self.capacity = int(capacity)
        if self.capacity <= 0:
            raise ValueError("capacity must be positive")
        self.state = np.zeros((self.capacity, int(state_dim)), dtype=np.float64)
        self.action = np.zeros((self.capacity, int(action_dim)), dtype=np.float64)
        self.costs = np.zeros((self.capacity, int(cost_dim)), dtype=np.float64)
        self.next_state = np.zeros((self.capacity, int(state_dim)), dtype=np.float64)
        self.count = 0
        self.write_index = 0

    def store(self, state, action, costs, next_state):
        self.state[self.write_index] = np.asarray(state, dtype=np.float64).reshape(-1)
        self.action[self.write_index] = np.asarray(action, dtype=np.float64).reshape(-1)
        self.costs[self.write_index] = np.asarray(costs, dtype=np.float64).reshape(-1)
        self.next_state[self.write_index] = np.asarray(next_state, dtype=np.float64).reshape(-1)
        self.write_index = (self.write_index + 1) % self.capacity
        self.count = min(self.count + 1, self.capacity)

    def __len__(self):
        return int(self.count)

    def arrays(self):
        if self.count < self.capacity:
            sl = slice(0, self.count)
            return self.state[sl], self.action[sl], self.costs[sl], self.next_state[sl]
        order = np.concatenate((np.arange(self.write_index, self.capacity), np.arange(0, self.write_index)))
        return self.state[order], self.action[order], self.costs[order], self.next_state[order]

    def latest(self, batch_size):
        states, actions, costs, next_states = self.arrays()
        batch_size = min(int(batch_size), states.shape[0])
        return states[-batch_size:], actions[-batch_size:], costs[-batch_size:], next_states[-batch_size:]
