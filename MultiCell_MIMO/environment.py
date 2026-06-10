import numpy as np


DEFAULT_DIRECT_GAIN_DB_RANGE = (-10.0, 10.0)
DEFAULT_CROSS_GAIN_DB_RANGE = (-30.0, -10.0)
DEFAULT_PATH_COUNT = 4
ACTION_EPS = 1e-6


class MultiCellMIMOEnv:
    def __init__(
        self,
        seed,
        nt,
        cell_count,
        users_per_cell,
        arrival_upper=2.0,
        queue_max=5.0,
        direct_gain_db_range=DEFAULT_DIRECT_GAIN_DB_RANGE,
        cross_gain_db_range=DEFAULT_CROSS_GAIN_DB_RANGE,
        path_count=DEFAULT_PATH_COUNT,
    ):
        self.seed = int(seed)
        self.nt = int(nt)
        self.cell_count = int(cell_count)
        self.users_per_cell = int(users_per_cell)
        if self.nt <= 0 or self.cell_count <= 0 or self.users_per_cell <= 0:
            raise ValueError("nt, cell_count and users_per_cell must be positive")

        self.user_count = self.cell_count * self.users_per_cell
        self.constraint_dim = self.user_count
        self.cell_action_dim = self.users_per_cell + 1
        self.action_dim = self.cell_count * self.cell_action_dim
        self.channel_size = self.cell_count * self.users_per_cell * self.cell_count * self.nt
        self.state_dim = 2 * self.channel_size + self.user_count
        self.local_actor_state_dim = 2 * self.users_per_cell * self.nt + self.users_per_cell
        self.local_critic_state_dim = 2 * self.users_per_cell * self.cell_count * self.nt + self.users_per_cell
        self.arrival_upper = float(arrival_upper)
        self.queue_max = float(queue_max)
        self.noise_power = 1e-6
        self.path_count = int(path_count)
        self.rng = self._make_rng(self.seed)
        init_rng = self._make_rng(self.seed)

        self.path_gain = self._build_path_gain(init_rng, direct_gain_db_range, cross_gain_db_range)
        self.alpha_power = self._build_alpha_power(init_rng)
        self.array_response = self._build_array_response(init_rng)
        self.h = np.zeros((self.cell_count, self.users_per_cell, self.cell_count, self.nt), dtype=np.complex128)
        self.queue = np.zeros((self.cell_count, self.users_per_cell), dtype=np.float64)
        self.state = np.zeros((self.state_dim,), dtype=np.float64)

    def _make_rng(self, seed):
        return np.random.RandomState(int(seed))

    def _build_path_gain(self, rng, direct_range, cross_range):
        path_gain_db = np.zeros((self.cell_count, self.users_per_cell, self.cell_count), dtype=np.float64)
        for rx_cell in range(self.cell_count):
            for user in range(self.users_per_cell):
                for tx_cell in range(self.cell_count):
                    low, high = direct_range if rx_cell == tx_cell else cross_range
                    path_gain_db[rx_cell, user, tx_cell] = rng.uniform(low, high)
        return 10.0 ** (path_gain_db / 10.0)

    def _build_alpha_power(self, rng):
        alpha = np.zeros((self.cell_count, self.users_per_cell, self.cell_count, self.path_count), dtype=np.float64)
        for rx_cell in range(self.cell_count):
            for user in range(self.users_per_cell):
                for tx_cell in range(self.cell_count):
                    raw = rng.exponential(scale=1.0, size=self.path_count)
                    alpha[rx_cell, user, tx_cell] = raw * self.path_gain[rx_cell, user, tx_cell] / np.sum(raw)
        return alpha

    def _laplace_angle(self, rng, mu=0.0, angular_spread=5.0):
        b = angular_spread / np.sqrt(2.0)
        a = rng.rand() - 0.5
        return float(mu - b * np.sign(a) * np.log(1.0 - 2.0 * np.abs(a)))

    def _build_array_response(self, rng):
        response = np.zeros(
            (self.cell_count, self.users_per_cell, self.cell_count, self.nt, self.path_count),
            dtype=np.complex128,
        )
        for rx_cell in range(self.cell_count):
            for user in range(self.users_per_cell):
                for tx_cell in range(self.cell_count):
                    for path_idx in range(self.path_count):
                        aod = self._laplace_angle(rng)
                        response[rx_cell, user, tx_cell, :, path_idx] = np.exp(
                            1j * np.pi * np.sin(aod) * np.arange(self.nt)
                        )
        return response

    def _refresh_channels(self):
        for rx_cell in range(self.cell_count):
            for user in range(self.users_per_cell):
                for tx_cell in range(self.cell_count):
                    alpha_power = self.alpha_power[rx_cell, user, tx_cell]
                    alpha = (
                        np.sqrt(alpha_power / 2.0) * self.rng.randn(self.path_count)
                        + 1j * np.sqrt(alpha_power / 2.0) * self.rng.randn(self.path_count)
                    )
                    self.h[rx_cell, user, tx_cell] = self.array_response[rx_cell, user, tx_cell] @ alpha

    def _compose_state(self):
        self.state = np.hstack((np.real(self.h).reshape(-1), np.imag(self.h).reshape(-1), self.queue.reshape(-1)))
        return self.state.astype(np.float64, copy=False)

    def reset(self):
        self.rng = self._make_rng(self.seed)
        self.queue = np.zeros((self.cell_count, self.users_per_cell), dtype=np.float64)
        self._refresh_channels()
        return self._compose_state()

    def _decode_action(self, action):
        action = np.asarray(action, dtype=np.float64).reshape(-1)
        if action.size != self.action_dim:
            raise ValueError("action_dim mismatch: expected {0}, got {1}".format(self.action_dim, action.size))
        action = np.maximum(action, ACTION_EPS)
        action = action.reshape(self.cell_count, self.cell_action_dim)
        power = action[:, : self.users_per_cell]
        reg = action[:, self.users_per_cell]
        return power, reg

    def _cell_beamformer(self, tx_cell, reg_value):
        h_direct = self.h[tx_cell, :, tx_cell, :]
        eye = np.eye(self.users_per_cell)
        gram = h_direct @ h_direct.conjugate().T + float(reg_value) * eye
        try:
            beamformer = h_direct.conjugate().T @ np.linalg.inv(gram)
        except np.linalg.LinAlgError:
            beamformer = h_direct.conjugate().T @ np.linalg.pinv(gram)
        norms = np.linalg.norm(beamformer, axis=0) + 1e-7
        return beamformer @ np.diag(1.0 / norms)

    def _compute_rates(self, power, reg):
        beamformers = np.zeros((self.cell_count, self.nt, self.users_per_cell), dtype=np.complex128)
        for tx_cell in range(self.cell_count):
            beamformers[tx_cell] = self._cell_beamformer(tx_cell, reg[tx_cell])

        rates = np.zeros((self.cell_count, self.users_per_cell), dtype=np.float64)
        for rx_cell in range(self.cell_count):
            for user in range(self.users_per_cell):
                desired = self.h[rx_cell, user, rx_cell]
                desired_gain = np.abs(desired @ beamformers[rx_cell, :, user]) ** 2
                numerator = float(power[rx_cell, user]) * desired_gain
                interference = 0.0
                for tx_cell in range(self.cell_count):
                    link = self.h[rx_cell, user, tx_cell]
                    gains = np.abs(link @ beamformers[tx_cell]) ** 2
                    for stream in range(self.users_per_cell):
                        if tx_cell == rx_cell and stream == user:
                            continue
                        interference += float(power[tx_cell, stream]) * gains[stream]
                rates[rx_cell, user] = np.log2(1.0 + numerator / (interference + self.noise_power))
        return rates

    def step(self, action):
        power, reg = self._decode_action(action)
        objective_cost = float(np.sum(power))
        current_costs = self.queue.reshape(-1).copy()
        info = {"cost_{0}".format(idx + 1): float(value) for idx, value in enumerate(current_costs)}
        info["cost"] = float(np.sum(current_costs))
        info["cell_cost"] = np.sum(self.queue, axis=1).astype(np.float64, copy=False)

        rates = self._compute_rates(power, reg)
        arrivals = self.rng.uniform(0.0, self.arrival_upper, size=(self.cell_count, self.users_per_cell))
        self.queue = np.clip(self.queue + arrivals - rates, 0.0, self.queue_max)
        self._refresh_channels()
        return self._compose_state(), objective_cost, False, info

    def _split_state(self, state):
        state = np.asarray(state, dtype=np.float64).reshape(-1)
        h_real = state[: self.channel_size].reshape(self.cell_count, self.users_per_cell, self.cell_count, self.nt)
        h_imag = state[self.channel_size : 2 * self.channel_size].reshape(
            self.cell_count, self.users_per_cell, self.cell_count, self.nt
        )
        queue = state[2 * self.channel_size :].reshape(self.cell_count, self.users_per_cell)
        return h_real, h_imag, queue

    def local_actor_observations_from_state(self, state):
        h_real, h_imag, queue = self._split_state(state)
        blocks = []
        for cell in range(self.cell_count):
            blocks.append(
                np.hstack(
                    (
                        h_real[cell, :, cell, :].reshape(-1),
                        h_imag[cell, :, cell, :].reshape(-1),
                        queue[cell].reshape(-1),
                    )
                )
            )
        return np.asarray(blocks, dtype=np.float64)

    def local_critic_observations_from_state(self, state):
        h_real, h_imag, queue = self._split_state(state)
        blocks = []
        for cell in range(self.cell_count):
            blocks.append(
                np.hstack(
                    (
                        h_real[cell, :, :, :].reshape(-1),
                        h_imag[cell, :, :, :].reshape(-1),
                        queue[cell].reshape(-1),
                    )
                )
            )
        return np.asarray(blocks, dtype=np.float64)

    def batch_local_actor_observations(self, states):
        return np.stack([self.local_actor_observations_from_state(state) for state in np.asarray(states)], axis=0)

    def batch_local_critic_observations(self, states):
        return np.stack([self.local_critic_observations_from_state(state) for state in np.asarray(states)], axis=0)

    def local_actor_observations(self):
        return self.local_actor_observations_from_state(self.state)

    def local_critic_observations(self):
        return self.local_critic_observations_from_state(self.state)
