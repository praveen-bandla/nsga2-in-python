import math

import numpy as np

from nsga2.individual import Individual
from nsga2.problem import Problem


class PortfolioProblem(Problem):
    def __init__(self, mean_returns, cov_matrix, std_returns, risk_free_rate=0.0):
        self.mean_returns_np = mean_returns
        self.cov_matrix_np = cov_matrix
        self.std_returns_np = std_returns
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(mean_returns)

        super().__init__(
            objectives=[self._dummy_objective],
            num_of_variables=self.num_assets,
            variables_range=[(0.0, 1.0) for _ in range(self.num_assets)],
            expand=False,
            same_range=False,
        )

    def _dummy_objective(self, weights):
        return 0.0

    def generate_individual(self):
        individual = Individual()
        weights = np.random.random(self.num_assets)
        weights /= weights.sum()
        individual.features = weights.astype(np.float64)
        return individual

    def repair_weights(self, individual):
        weights = np.maximum(individual.features, 0.0)
        total = weights.sum()
        if total == 0:
            weights = np.ones(self.num_assets) / self.num_assets
        else:
            weights /= total
        individual.features = weights

    def calculate_objectives(self, individual):
        self.repair_weights(individual)
        weights = individual.features

        cov_w = self.cov_matrix_np @ weights
        port_var = float(weights @ cov_w)
        port_return = float(weights @ self.mean_returns_np)
        weighted_vol = float(weights @ self.std_returns_np)
        port_std = math.sqrt(max(port_var, 0.0))

        if port_std < 1e-12:
            neg_sharpe = 0.0
            neg_div = 0.0
        else:
            neg_sharpe = -(port_return - self.risk_free_rate) / port_std
            neg_div = -weighted_vol / port_std

        individual.objectives = [neg_sharpe, port_var, neg_div]
