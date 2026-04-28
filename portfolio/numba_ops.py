"""
Numba JIT-compiled genetic operators for NSGA-II.
Used for crossover and mutation (per-gene loops with RNG).
"""

import numpy as np
from numba import njit


@njit(cache=True)
def sbx_crossover_jit(parent1, parent2, eta_c):
    """
    Simulated Binary Crossover on two weight vectors.
    """
    n = len(parent1)
    child1 = np.empty(n)
    child2 = np.empty(n)
    for i in range(n):
        u = np.random.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta_c + 1.0))
        else:
            beta = (2.0 * (1.0 - u)) ** (-1.0 / (eta_c + 1.0))
        midpoint = (parent1[i] + parent2[i]) * 0.5
        spread = abs(parent1[i] - parent2[i]) * 0.5
        child1[i] = midpoint + beta * spread
        child2[i] = midpoint - beta * spread
    return child1, child2


@njit(cache=True)
def gaussian_mutate_jit(features, mutation_prob, mutation_sigma):
    """
    Gaussian mutation with per-gene probability.
    """
    n = len(features)
    for i in range(n):
        if np.random.random() < mutation_prob:
            features[i] += np.random.normal(0.0, mutation_sigma)
            if features[i] < 0.0:
                features[i] = 0.0
            elif features[i] > 1.0:
                features[i] = 1.0
    return features
