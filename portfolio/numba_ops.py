"""
Numba JIT-compiled operations for NSGA-II portfolio optimization.

Compiles the tight inner loops that dominate runtime after NumPy handles
matrix math: non-dominated sorting, crossover, and mutation.
"""

import numpy as np
from numba import njit


@njit(cache=True)
def fast_nondominated_sort_jit(objectives):
    """
    Non-dominated sorting on a 2D objectives array.

    Args:
        objectives: (N, M) array, N=population size, M=num objectives.
                    All objectives are minimized.

    Returns:
        ranks: (N,) int array with Pareto front rank for each individual.
    """
    n = objectives.shape[0]
    m = objectives.shape[1]

    domination_count = np.zeros(n, dtype=np.int32)
    ranks = np.full(n, -1, dtype=np.int32)

    # Build domination matrix: dom[i,j] = True if i dominates j
    dom_matrix = np.zeros((n, n), dtype=np.bool_)
    for i in range(n):
        for j in range(i + 1, n):
            # Check if i dominates j
            i_dom_j = True
            i_any_better = False
            j_dom_i = True
            j_any_better = False
            for k in range(m):
                if objectives[i, k] > objectives[j, k]:
                    i_dom_j = False
                if objectives[i, k] < objectives[j, k]:
                    i_any_better = True
                if objectives[j, k] > objectives[i, k]:
                    j_dom_i = False
                if objectives[j, k] < objectives[i, k]:
                    j_any_better = True

            if i_dom_j and i_any_better:
                dom_matrix[i, j] = True
                domination_count[j] += 1
            elif j_dom_i and j_any_better:
                dom_matrix[j, i] = True
                domination_count[i] += 1

    # Peel off fronts layer by layer
    current_rank = 0
    assigned = 0
    while assigned < n:
        # Find all unassigned individuals with domination_count == 0
        found_any = False
        for i in range(n):
            if ranks[i] == -1 and domination_count[i] == 0:
                ranks[i] = current_rank
                assigned += 1
                found_any = True

        # Decrement counts for solutions dominated by this front
        for i in range(n):
            if ranks[i] == current_rank:
                for j in range(n):
                    if dom_matrix[i, j] and ranks[j] == -1:
                        domination_count[j] -= 1

        current_rank += 1
        if not found_any:
            for i in range(n):
                if ranks[i] == -1:
                    ranks[i] = current_rank
                    assigned += 1
            break

    return ranks


@njit(cache=True)
def sbx_crossover_jit(parent1, parent2, eta_c):
    """
    Simulated Binary Crossover on two weight vectors.

    Args:
        parent1, parent2: (N,) float arrays of portfolio weights.
        eta_c: distribution index (higher = children closer to parents).

    Returns:
        child1, child2: (N,) float arrays.
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

    Args:
        features: (N,) float array of portfolio weights (modified in-place).
        mutation_prob: probability of mutating each gene.
        mutation_sigma: std dev of Gaussian noise.

    Returns:
        features: the mutated array (same reference, modified in-place).
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
