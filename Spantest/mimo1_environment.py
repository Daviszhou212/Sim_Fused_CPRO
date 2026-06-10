from __future__ import annotations

import numpy as np


class EnvironmentMIMO:
    """MIMO1 power-allocation environment copied into Spantest for isolation."""

    def __init__(self, seed: int, nt: int = 8, ue_num: int = 4):
        self.seed = int(seed)
        self.seed_step = int(seed)
        self.nt = int(nt)
        self.ue_num = int(ue_num)
        self.user_per_group = 2
        self.group_num = int(self.ue_num / self.user_per_group)
        self.state_dim = 2 * self.ue_num * self.nt + self.ue_num
        self.action_dim = self.ue_num + 1
        self.np_paths = 4
        rng_state = np.random.get_state()
        np.random.seed(self.seed)
        path_gain_db = np.random.uniform(-10, 10, self.group_num)
        self.path_gain = 10 ** (path_gain_db / 10)
        alpha_power_group = np.zeros((self.group_num, self.np_paths))
        for group in range(self.group_num):
            tmp = np.random.exponential(scale=1, size=self.np_paths)
            alpha_power_group[group] = (tmp * self.path_gain[group]) / np.sum(tmp)
        self.alpha_power = np.tile(alpha_power_group, (self.user_per_group, 1))
        array_group = np.zeros((self.group_num * self.nt, self.np_paths), dtype=np.complex128)
        for group in range(self.group_num):
            a_tmp = np.zeros((self.nt, self.np_paths), dtype=np.complex128)
            for idx in range(self.np_paths):
                aod = self.laprnd(mu=0, angular_spread=5)
                a_tmp[:, idx] = np.exp(1j * np.pi * np.sin(aod) * np.arange(0, self.nt))
            array_group[group * self.nt : (group + 1) * self.nt] = a_tmp
        self.array_response = np.tile(array_group, (self.user_per_group, 1))
        np.random.set_state(rng_state)

        self.h_g = np.zeros((self.group_num, self.nt), dtype=np.complex128)
        self.h = np.zeros((self.ue_num, self.nt), dtype=np.complex128)
        self.delay = np.zeros(self.ue_num)
        self.state = np.zeros(self.state_dim)
        self.noise_power = 1e-6
        self.delay_max = 5

    def reset(self):
        np.random.seed(self.seed)
        for group in range(self.group_num):
            alpha_power_g = self.alpha_power[group]
            a_g = self.array_response[group * self.nt : (group + 1) * self.nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.np_paths) + 1j * np.sqrt(
                alpha_power_g / 2
            ) * np.random.randn(self.np_paths)
            self.h_g[group] = a_g @ alpha_g
        self.h = np.repeat(self.h_g, self.user_per_group, axis=0)
        self.delay = np.zeros(self.ue_num)
        self.state = self._state_vector()
        return self.state

    def step(self, action):
        np.random.seed(self.seed_step)
        self.seed_step += 1
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        action[action <= 0] = 1e-6
        power = action[: self.ue_num]
        reg_factor = action[self.ue_num]

        objective_cost = float(np.sum(power))
        costs = self.delay.copy()
        info = {"cost_" + str(i): float(costs[i - 1]) for i in range(1, self.ue_num + 1)}
        info["cost"] = float(np.sum(costs))

        try:
            v = self.h.conjugate().T @ np.linalg.inv(
                self.h @ self.h.conjugate().T + reg_factor * np.eye(self.ue_num)
            )
        except np.linalg.LinAlgError:
            v = self.h.conjugate().T @ np.linalg.pinv(
                self.h @ self.h.conjugate().T + reg_factor * np.eye(self.ue_num)
            )
        norm_vector = np.zeros(self.ue_num)
        for user in range(self.ue_num):
            norm_vector[user] = 1 / (np.linalg.norm(v[:, user]) + 1e-7)
        v_tilda = v @ np.diag(norm_vector)

        hv_tilda = self.h @ v_tilda
        rates = np.zeros(self.ue_num)
        for user in range(self.ue_num):
            module_sq = np.abs(hv_tilda[user]) ** 2
            numerator = power[user] * module_sq[user]
            module_sq[user] = 0
            denominator = np.sum(power * module_sq) + self.noise_power
            rates[user] = np.log2(1 + numerator / denominator)
        arrivals = np.random.uniform(0, 2, self.ue_num)
        self.delay = np.clip(self.delay + arrivals - rates, 0.0, self.delay_max)

        for group in range(self.group_num):
            alpha_power_g = self.alpha_power[group]
            a_g = self.array_response[group * self.nt : (group + 1) * self.nt]
            alpha_g = np.sqrt(alpha_power_g / 2) * np.random.randn(self.np_paths) + 1j * np.sqrt(
                alpha_power_g / 2
            ) * np.random.randn(self.np_paths)
            self.h_g[group] = a_g @ alpha_g
        self.h = np.repeat(self.h_g, self.user_per_group, axis=0)
        self.state = self._state_vector()
        return self.state, objective_cost, False, info

    def _state_vector(self):
        return np.hstack((np.real(self.h).reshape(-1), np.imag(self.h).reshape(-1), self.delay))

    @staticmethod
    def laprnd(mu, angular_spread):
        b = angular_spread / np.sqrt(2)
        a = np.random.rand(1) - 0.5
        return mu - b * np.sign(a) * np.log(1 - 2 * np.abs(a))

