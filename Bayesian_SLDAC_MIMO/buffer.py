import numpy as np


class DataStorage(object):
    def __init__(self, T, num_new_data, state_dim, action_dim, constraint_dim, window, q):
        self.T = int(T)
        self.num_new_data = int(num_new_data / q)
        self.state_memory = np.zeros((2 * self.T, state_dim))
        self.action_memory = np.zeros((2 * self.T, action_dim))
        self.cost_memory = np.zeros((2 * self.T, 1 + constraint_dim))
        self.next_state_memory = np.zeros((2 * self.T, state_dim))
        self.state_memory_tmp = np.zeros((self.num_new_data, state_dim))
        self.action_memory_tmp = np.zeros((self.num_new_data, action_dim))
        self.cost_memory_tmp = np.zeros((self.num_new_data, 1 + constraint_dim))
        self.next_state_memory_tmp = np.zeros((self.num_new_data, state_dim))
        self.count = 0
        self.window = int(window)
        self.aver_reward_memory = np.zeros((self.window, 1))
        self.aver_cost_memory = np.zeros((self.window, 1))
        self.aver_reward_memory_tmp = np.zeros((self.num_new_data, 1))
        self.aver_cost_memory_tmp = np.zeros((self.num_new_data, 1))

    def store_experiences(self, state, action, costs, next_state, aver_reward, aver_cost):
        if self.count < 2 * self.T:
            self.state_memory[self.count] = state
            self.action_memory[self.count] = action
            self.cost_memory[self.count] = costs
            self.next_state_memory[self.count] = next_state
        else:
            index = self.count % self.num_new_data
            self.state_memory_tmp[index] = state
            self.action_memory_tmp[index] = action
            self.cost_memory_tmp[index] = costs
            self.next_state_memory_tmp[index] = next_state
            if index == self.num_new_data - 1:
                self.state_memory[0 : 2 * self.T - self.num_new_data] = self.state_memory[self.num_new_data :]
                self.state_memory[2 * self.T - self.num_new_data :] = self.state_memory_tmp
                self.action_memory[0 : 2 * self.T - self.num_new_data] = self.action_memory[self.num_new_data :]
                self.action_memory[2 * self.T - self.num_new_data :] = self.action_memory_tmp
                self.cost_memory[0 : 2 * self.T - self.num_new_data] = self.cost_memory[self.num_new_data :]
                self.cost_memory[2 * self.T - self.num_new_data :] = self.cost_memory_tmp
                self.next_state_memory[0 : 2 * self.T - self.num_new_data] = self.next_state_memory[self.num_new_data :]
                self.next_state_memory[2 * self.T - self.num_new_data :] = self.next_state_memory_tmp

        if self.count < self.window:
            self.aver_reward_memory[self.count] = aver_reward
            self.aver_cost_memory[self.count] = aver_cost
        else:
            index = self.count % self.num_new_data
            self.aver_reward_memory_tmp[index] = aver_reward
            self.aver_cost_memory_tmp[index] = aver_cost
            if index == self.num_new_data - 1:
                self.aver_reward_memory[0 : self.window - self.num_new_data] = self.aver_reward_memory[self.num_new_data :]
                self.aver_reward_memory[self.window - self.num_new_data :] = self.aver_reward_memory_tmp
                self.aver_cost_memory[0 : self.window - self.num_new_data] = self.aver_cost_memory[self.num_new_data :]
                self.aver_cost_memory[self.window - self.num_new_data :] = self.aver_cost_memory_tmp
        self.count += 1

    def take_experiences(self):
        if self.count < self.window:
            return (
                self.state_memory,
                self.action_memory,
                self.cost_memory,
                self.next_state_memory,
                self.aver_reward_memory[0 : self.count],
                self.aver_cost_memory[0 : self.count],
            )
        return (
            self.state_memory,
            self.action_memory,
            self.cost_memory,
            self.next_state_memory,
            self.aver_reward_memory,
            self.aver_cost_memory,
        )

