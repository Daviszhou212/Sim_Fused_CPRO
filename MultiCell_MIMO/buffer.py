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


class LegacySLDACBuffer:
    def __init__(self, t_horizon, num_new_data, state_dim, action_dim, cost_dim, window):
        self.t_horizon = int(t_horizon)
        self.num_new_data = int(num_new_data)
        self.window = int(window)
        if self.t_horizon <= 0 or self.num_new_data <= 0 or self.window <= 0:
            raise ValueError("t_horizon, num_new_data and window must be positive")
        if 2 * self.t_horizon < self.num_new_data:
            raise ValueError("2 * t_horizon must be at least num_new_data")
        if self.window < self.num_new_data:
            raise ValueError("window must be at least num_new_data")

        self.count = 0
        self.state_memory = np.zeros((2 * self.t_horizon, int(state_dim)), dtype=np.float64)
        self.action_memory = np.zeros((2 * self.t_horizon, int(action_dim)), dtype=np.float64)
        self.cost_memory = np.zeros((2 * self.t_horizon, int(cost_dim)), dtype=np.float64)
        self.next_state_memory = np.zeros((2 * self.t_horizon, int(state_dim)), dtype=np.float64)

        self.state_memory_tmp = np.zeros((self.num_new_data, int(state_dim)), dtype=np.float64)
        self.action_memory_tmp = np.zeros((self.num_new_data, int(action_dim)), dtype=np.float64)
        self.cost_memory_tmp = np.zeros((self.num_new_data, int(cost_dim)), dtype=np.float64)
        self.next_state_memory_tmp = np.zeros((self.num_new_data, int(state_dim)), dtype=np.float64)

        self.aver_objective_memory = np.zeros((self.window, 1), dtype=np.float64)
        self.aver_cost_memory = np.zeros((self.window, 1), dtype=np.float64)
        self.aver_objective_memory_tmp = np.zeros((self.num_new_data, 1), dtype=np.float64)
        self.aver_cost_memory_tmp = np.zeros((self.num_new_data, 1), dtype=np.float64)

    def _store_training_window(self, state, action, costs, next_state):
        if self.count < 2 * self.t_horizon:
            self.state_memory[self.count] = state
            self.action_memory[self.count] = action
            self.cost_memory[self.count] = costs
            self.next_state_memory[self.count] = next_state
            return

        index = self.count % self.num_new_data
        self.state_memory_tmp[index] = state
        self.action_memory_tmp[index] = action
        self.cost_memory_tmp[index] = costs
        self.next_state_memory_tmp[index] = next_state
        if index != self.num_new_data - 1:
            return

        keep_count = 2 * self.t_horizon - self.num_new_data
        self.state_memory[:keep_count] = self.state_memory[self.num_new_data :]
        self.state_memory[keep_count:] = self.state_memory_tmp
        self.action_memory[:keep_count] = self.action_memory[self.num_new_data :]
        self.action_memory[keep_count:] = self.action_memory_tmp
        self.cost_memory[:keep_count] = self.cost_memory[self.num_new_data :]
        self.cost_memory[keep_count:] = self.cost_memory_tmp
        self.next_state_memory[:keep_count] = self.next_state_memory[self.num_new_data :]
        self.next_state_memory[keep_count:] = self.next_state_memory_tmp

    def _store_average_window(self, aver_objective, aver_cost):
        if self.count < self.window:
            self.aver_objective_memory[self.count] = aver_objective
            self.aver_cost_memory[self.count] = aver_cost
            return

        index = self.count % self.num_new_data
        self.aver_objective_memory_tmp[index] = aver_objective
        self.aver_cost_memory_tmp[index] = aver_cost
        if index != self.num_new_data - 1:
            return

        keep_count = self.window - self.num_new_data
        self.aver_objective_memory[:keep_count] = self.aver_objective_memory[self.num_new_data :]
        self.aver_objective_memory[keep_count:] = self.aver_objective_memory_tmp
        self.aver_cost_memory[:keep_count] = self.aver_cost_memory[self.num_new_data :]
        self.aver_cost_memory[keep_count:] = self.aver_cost_memory_tmp

    def store_experiences(self, state, action, costs, next_state, aver_objective, aver_cost):
        self._store_training_window(
            np.asarray(state, dtype=np.float64).reshape(-1),
            np.asarray(action, dtype=np.float64).reshape(-1),
            np.asarray(costs, dtype=np.float64).reshape(-1),
            np.asarray(next_state, dtype=np.float64).reshape(-1),
        )
        self._store_average_window(float(aver_objective), float(aver_cost))
        self.count += 1

    def take_experiences(self):
        if self.count < self.window:
            average_slice = slice(0, self.count)
            return (
                self.state_memory,
                self.action_memory,
                self.cost_memory,
                self.next_state_memory,
                self.aver_objective_memory[average_slice].reshape(-1),
                self.aver_cost_memory[average_slice].reshape(-1),
            )
        return (
            self.state_memory,
            self.action_memory,
            self.cost_memory,
            self.next_state_memory,
            self.aver_objective_memory.reshape(-1),
            self.aver_cost_memory.reshape(-1),
        )

    def __len__(self):
        return int(self.count)
