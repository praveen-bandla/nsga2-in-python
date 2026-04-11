"""
Portfolio optimization problem for NSGA-II.
Three objectives: Sharpe ratio, variance, diversification ratio.
All minimized (Sharpe and diversification are negated).

Uses NumPy for matrix operations (standard practice).
"""

import random
import math
import numpy as np
from nsga2.problem import Problem
from nsga2.individual import Individual


class PortfolioProblem(Problem):
    """
    Multi-objective portfolio optimization problem.
    Handles weight constraints: non-negative, sum to 1.
    """

    def __init__(self, mean_returns, cov_matrix, std_returns, risk_free_rate=0.0):
        self.mean_returns_np = np.asarray(mean_returns, dtype=np.float64)
        self.cov_matrix_np = np.asarray(cov_matrix, dtype=np.float64)
        self.std_returns_np = np.asarray(std_returns, dtype=np.float64)
        self.risk_free_rate = risk_free_rate
        self.num_assets = len(mean_returns)

        objectives = [self._dummy_objective]
        variables_range = [(0.0, 1.0) for _ in range(self.num_assets)]

        super().__init__(
            objectives=objectives,
            num_of_variables=self.num_assets,
            variables_range=variables_range,
            expand=False,
            same_range=False,
        )

    def _dummy_objective(self, weights):
        return 0.0

    def generate_individual(self):
        """Generate a random portfolio with weights summing to 1."""
        individual = Individual()
        weights = np.random.random(self.num_assets)
        weights /= weights.sum()
        individual.features = weights.tolist()
        return individual

    def repair_weights(self, individual):
        """Clip negatives to zero and normalize to sum to 1."""
        w = np.array(individual.features, dtype=np.float64)
        np.maximum(w, 0.0, out=w)
        total = w.sum()
        if total == 0:
            w[:] = 1.0 / self.num_assets
        else:
            w /= total
        individual.features = w.tolist()

    def calculate_objectives(self, individual):
        """Repair weights and compute all 3 objectives in one pass."""
        self.repair_weights(individual)
        w = np.array(individual.features, dtype=np.float64)

        cov_w = self.cov_matrix_np @ w
        port_var = float(w @ cov_w)
        port_return = float(w @ self.mean_returns_np)
        weighted_vol = float(w @ self.std_returns_np)
        port_std = math.sqrt(max(port_var, 0.0))

        if port_std < 1e-12:
            neg_sharpe = 0.0
            neg_div = 0.0
        else:
            neg_sharpe = -(port_return - self.risk_free_rate) / port_std
            neg_div = -weighted_vol / port_std

        individual.objectives = [neg_sharpe, port_var, neg_div]
