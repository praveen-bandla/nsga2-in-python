"""
Portfolio optimization problem for NSGA-II.
Three objectives: Sharpe ratio, variance, diversification ratio.
All minimized (Sharpe and diversification are negated).
"""

import random
import math
from nsga2.problem import Problem
from nsga2.individual import Individual


class PortfolioProblem(Problem):
    """
    Multi-objective portfolio optimization problem.
    Handles weight constraints: non-negative, sum to 1.
    """

    def __init__(self, mean_returns, cov_matrix, std_returns, risk_free_rate=0.0):
        self.mean_returns = list(mean_returns)
        self.cov_matrix = [list(row) for row in cov_matrix]
        self.std_returns = list(std_returns)
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(mean_returns)

        objectives = [
            self.neg_sharpe_ratio,
            self.portfolio_variance,
            self.neg_diversification_ratio,
        ]
        variables_range = [(0.0, 1.0) for _ in range(self.num_assets)]

        super().__init__(
            objectives=objectives,
            num_of_variables=self.num_assets,
            variables_range=variables_range,
            expand=False,
            same_range=False,
        )

    def generate_individual(self):
        """Generate a random portfolio with weights summing to 1."""
        individual = Individual()
        weights = [random.random() for _ in range(self.num_assets)]
        total = sum(weights)
        individual.features = [w / total for w in weights]
        return individual

    def repair_weights(self, individual):
        """Clip negatives to zero and normalize to sum to 1."""
        features = [max(0.0, f) for f in individual.features]
        total = sum(features)
        if total == 0:
            features = [1.0 / self.num_assets] * self.num_assets
        else:
            features = [f / total for f in features]
        individual.features = features

    def calculate_objectives(self, individual):
        """Repair weights then compute all objectives."""
        self.repair_weights(individual)
        individual.objectives = [f(individual.features) for f in self.objectives]

    # -- Pure Python math (no numpy) -- baseline for optimization later --

    def _dot(self, a, b):
        s = 0.0
        for x, y in zip(a, b):
            s += x * y
        return s

    def _mat_vec(self, mat, vec):
        result = []
        for row in mat:
            s = 0.0
            for m, v in zip(row, vec):
                s += m * v
            result.append(s)
        return result

    def _portfolio_return(self, weights):
        return self._dot(weights, self.mean_returns)

    def _portfolio_variance_value(self, weights):
        cov_w = self._mat_vec(self.cov_matrix, weights)
        return self._dot(weights, cov_w)

    def neg_sharpe_ratio(self, weights):
        """Negative Sharpe ratio. Sharpe = (E[r] - r_f) / std(r)."""
        port_return = self._portfolio_return(weights)
        port_var = self._portfolio_variance_value(weights)
        port_std = math.sqrt(max(port_var, 0.0))
        if port_std < 1e-12:
            return 0.0
        return -(port_return - self.risk_free_rate) / port_std

    def portfolio_variance(self, weights):
        """Portfolio variance: w^T @ Sigma @ w."""
        return self._portfolio_variance_value(weights)

    def neg_diversification_ratio(self, weights):
        """Negative diversification ratio. DR = (w . sigma) / sigma_p."""
        weighted_vol = self._dot(weights, self.std_returns)
        port_var = self._portfolio_variance_value(weights)
        port_std = math.sqrt(max(port_var, 0.0))
        if port_std < 1e-12:
            return 0.0
        return -weighted_vol / port_std
