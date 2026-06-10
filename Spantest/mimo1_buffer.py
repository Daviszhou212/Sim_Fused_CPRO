from __future__ import annotations

import numpy as np


class DataStorage:
    def __init__(self, t_horizon, num_new_data, state_dim, action_dim, constraint_dim, window):
        self.t_horizon = int(t_horizon)
        self.num_new_data = int(num_new_data)
        self.window = int(window)
        self.count = 0
        self.state_memory = np.zeros((2 * self.t_horizon, state_dim))
        self.action_memory = np.zeros((2 * self.t_horizon, action_dim))
        self.cost_memory = np.zeros((2 * self.t_horizon, 1 + constraint_dim))
        self.next_state_memory = np.zeros((2 * self.t_horizon, state_dim))
        self.state_tmp = np.zeros((self.num_new_data, state_dim))
        self.action_tmp = np.zeros((self.num_new_data, action_dim))
        self.cost_tmp = np.zeros((self.num_new_data, 1 + constraint_dim))
        self.next_state_tmp = np.zeros((self.num_new_data, state_dim))
        self.reward_window = np.zeros((self.window, 1))
        self.cost_window = np.zeros((self.window, 1))
        self.reward_tmp = np.zeros((self.num_new_data, 1))
        self.aver_cost_tmp = np.zeros((self.num_new_data, 1))

    def store(self, state, action, costs, next_state, objective_cost, avg_constraint_cost):
        if self.count < 2 * self.t_horizon:
            self.state_memory[self.count] = state
            self.action_memory[self.count] = action
            self.cost_memory[self.count] = costs
            self.next_state_memory[self.count] = next_state
        else:
            idx = self.count % self.num_new_data
            self.state_tmp[idx] = state
            self.action_tmp[idx] = action
            self.cost_tmp[idx] = costs
            self.next_state_tmp[idx] = next_state
            if idx == self.num_new_data - 1:
                keep = 2 * self.t_horizon - self.num_new_data
                self.state_memory[:keep] = self.state_memory[self.num_new_data :]
                self.state_memory[keep:] = self.state_tmp
                self.action_memory[:keep] = self.action_memory[self.num_new_data :]
                self.action_memory[keep:] = self.action_tmp
                self.cost_memory[:keep] = self.cost_memory[self.num_new_data :]
                self.cost_memory[keep:] = self.cost_tmp
                self.next_state_memory[:keep] = self.next_state_memory[self.num_new_data :]
                self.next_state_memory[keep:] = self.next_state_tmp

        if self.count < self.window:
            self.reward_window[self.count] = objective_cost
            self.cost_window[self.count] = avg_constraint_cost
        else:
            idx = self.count % self.num_new_data
            self.reward_tmp[idx] = objective_cost
            self.aver_cost_tmp[idx] = avg_constraint_cost
            if idx == self.num_new_data - 1:
                keep = self.window - self.num_new_data
                self.reward_window[:keep] = self.reward_window[self.num_new_data :]
                self.reward_window[keep:] = self.reward_tmp
                self.cost_window[:keep] = self.cost_window[self.num_new_data :]
                self.cost_window[keep:] = self.aver_cost_tmp
        self.count += 1

    def take(self):
        if self.count < self.window:
            return (
                self.state_memory,
                self.action_memory,
                self.cost_memory,
                self.next_state_memory,
                self.reward_window[: self.count],
                self.cost_window[: self.count],
            )
        return (
            self.state_memory,
            self.action_memory,
            self.cost_memory,
            self.next_state_memory,
            self.reward_window,
            self.cost_window,
        )

