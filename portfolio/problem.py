"""
Portfolio optimization problem for NSGA-II.
Three objectives: portfolio return, variance, diversification ratio.
All minimized (return and diversification are negated).
"""

import random
import math
from nsga2.problem import Problem
from nsga2.individual import Individual
import numpy as np


class PortfolioProblem(Problem):
    """
    Multi-objective portfolio optimization problem.
    Handles weight constraints: non-negative, sum to 1.
    """

    def __init__(self, mean_returns, cov_matrix, std_returns):
        # mod 1: storing as numpy arrays
        self.mean_returns = np.array(mean_returns)
        self.cov_matrix   = np.array(cov_matrix)
        self.std_returns  = np.array(std_returns)
        self.num_assets = len(mean_returns)

        objectives = [
            self.neg_return,
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

    # Vecotrizing repair weights
    def repair_weights(self, individual):
        """Clip negatives to zero and normalize to sum to 1."""
        features = np.maximum(0.0, np.array(individual.features))
        total = features.sum()
        if total == 0:
            features = np.ones(self.num_assets) / self.num_assets
        else:
            features = features / total 
        individual.features = features
        
    # -----

    def calculate_objectives(self, individual):
        """Repair weights then compute all objectives."""
        self.repair_weights(individual)
        individual.objectives = [f(individual.features) for f in self.objectives]

    # -- Pure Python math (no numpy) -- baseline for optimization later --

    # mod 2: using dot product vecotrization 
    def _dot(self, a, b):
        return float(np.dot(a, b))

    def _mat_vec(self, mat, vec):
        return np.dot(mat, vec)
    
    # ----

    def _portfolio_return(self, weights):
        return self._dot(weights, self.mean_returns)

    def _portfolio_variance_value(self, weights):
        cov_w = self._mat_vec(self.cov_matrix, weights)
        return self._dot(weights, cov_w)

    def neg_return(self, weights):
        """Negative portfolio return (negated so NSGA-II minimizes → maximizes return)."""
        return -self._portfolio_return(weights)

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
