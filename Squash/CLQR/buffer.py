import numpy as np
import random
from collections import deque

class DataStorage(object):

	def __init__(self, T, window, num_new_data, q, state_dim, action_dim, constraint_dim):
		self.T = T
		self.window=window
		self.q=q
		self.num_new_data = num_new_data
		self.count = 0
		self.state_memory = np.zeros((2 * self.T, state_dim))
		self.next_state_memory = np.zeros((2 * self.T, state_dim))
		self.action_memory = np.zeros((2 * self.T, action_dim))
		self.cost_memory = np.zeros((2 * self.T, 1+constraint_dim))
		self.state_memory_tmp = np.zeros((int(self.num_new_data/self.q), state_dim))
		self.action_memory_tmp = np.zeros((int(self.num_new_data/self.q), action_dim))
		self.cost_memory_tmp = np.zeros((int(self.num_new_data/self.q), 1+constraint_dim))
		self.next_state_memory_tmp = np.zeros((int(self.num_new_data/self.q), state_dim))

		self.aver_cost_memory = np.zeros((window, 1))
		self.aver_reward_memory = np.zeros((window, 1))
		self.aver_cost_memory_tmp = np.zeros((int(self.num_new_data / self.q), 1))
		self.aver_reward_memory_tmp = np.zeros((int(self.num_new_data / self.q), 1))



	def store_experiences(self, state, action, costs, next_state, aver_reward, aver_cost):
		if self.count < 2 * self.T:
			self.state_memory[self.count] = state
			self.action_memory[self.count] = action
			self.cost_memory[self.count] = costs
			self.next_state_memory[self.count] = next_state
		else:
			ind = self.count % (int(self.num_new_data/self.q))
			self.state_memory_tmp[ind] = state
			self.action_memory_tmp[ind] = action
			self.cost_memory_tmp[ind] = costs
			self.next_state_memory_tmp[ind] = next_state
			if ind == int(self.num_new_data/self.q)-1:
				self.state_memory[0: 2 * self.T - int(self.num_new_data/self.q)] = self.state_memory[int(self.num_new_data/self.q):]
				self.state_memory[2 * self.T - int(self.num_new_data/self.q):] = self.state_memory_tmp
				self.action_memory[0: 2 * self.T - int(self.num_new_data/self.q)] = self.action_memory[int(self.num_new_data/self.q):]
				self.action_memory[2 * self.T - int(self.num_new_data/self.q):] = self.action_memory_tmp
				self.cost_memory[0: 2 * self.T - int(self.num_new_data/self.q)] = self.cost_memory[int(self.num_new_data/self.q):]
				self.cost_memory[2 * self.T - int(self.num_new_data/self.q):] = self.cost_memory_tmp
				self.next_state_memory[0: 2 * self.T - int(self.num_new_data/self.q)] = self.next_state_memory[int(self.num_new_data/self.q):]
				self.next_state_memory[2 * self.T - int(self.num_new_data/self.q):] = self.next_state_memory_tmp

		if self.count < self.window:
			self.aver_cost_memory[self.count] = aver_cost
			self.aver_reward_memory[self.count] = aver_reward
		else:
			ind = self.count % (int(self.num_new_data / self.q))
			self.aver_cost_memory_tmp[ind] = aver_cost
			self.aver_reward_memory_tmp[ind] = aver_reward
			if ind == int(self.num_new_data / self.q) - 1:
				self.aver_cost_memory[0: self.window - int(self.num_new_data/self.q)] = self.aver_cost_memory[int(self.num_new_data/self.q):]
				self.aver_cost_memory[self.window - int(self.num_new_data/self.q):] = self.aver_cost_memory_tmp
				self.aver_reward_memory[0: self.window - int(self.num_new_data/self.q)] = self.aver_reward_memory[int(self.num_new_data/self.q):]
				self.aver_reward_memory[self.window - int(self.num_new_data/self.q):] = self.aver_reward_memory_tmp
		self.count += 1

	def take_experiences(self):
		if self.count>=self.window:
			return self.state_memory, self.action_memory, self.cost_memory, self.next_state_memory, self.aver_reward_memory, self.aver_cost_memory
		else:
			return self.state_memory, self.action_memory, self.cost_memory, self.next_state_memory, self.aver_reward_memory[0:self.count], self.aver_cost_memory[0:self.count]

