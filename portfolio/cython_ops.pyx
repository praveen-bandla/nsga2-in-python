# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-compiled operations for NSGA-II portfolio optimization.

Compiles the tight inner loops to C: non-dominated sorting,
SBX crossover, and Gaussian mutation.
"""

import numpy as np
cimport numpy as np
from libc.math cimport fabs, pow, sqrt
from libc.stdlib cimport rand, RAND_MAX, srand
from libc.time cimport time

np.import_array()


cdef double _rand_uniform() noexcept nogil:
    """Uniform random in [0, 1)."""
    return <double>rand() / (<double>RAND_MAX + 1.0)


cdef double _rand_normal(double mu, double sigma) noexcept nogil:
    """Box-Muller transform for Gaussian random."""
    cdef double u1, u2, z
    u1 = _rand_uniform()
    if u1 < 1e-15:
        u1 = 1e-15
    u2 = _rand_uniform()
    z = sqrt(-2.0 * log(u1)) * cos(6.283185307179586 * u2)
    return mu + sigma * z


cdef extern from "math.h" nogil:
    double log(double x)
    double cos(double x)


def fast_nondominated_sort_cy(np.ndarray[np.float64_t, ndim=2] objectives):
    """
    Non-dominated sorting on a 2D objectives array.

    Args:
        objectives: (N, M) array, all objectives minimized.

    Returns:
        ranks: (N,) int array with Pareto front rank for each individual.
    """
    cdef int n = objectives.shape[0]
    cdef int m = objectives.shape[1]
    cdef np.ndarray[np.int32_t, ndim=1] domination_count = np.zeros(n, dtype=np.int32)
    cdef np.ndarray[np.int32_t, ndim=1] ranks = np.full(n, -1, dtype=np.int32)
    cdef np.ndarray[np.uint8_t, ndim=2] dom_matrix = np.zeros((n, n), dtype=np.uint8)

    cdef int i, j, k
    cdef bint i_dom_j, i_any_better, j_dom_i, j_any_better
    cdef bint found_any
    cdef int current_rank, assigned

    # Build domination matrix
    for i in range(n):
        for j in range(i + 1, n):
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
                dom_matrix[i, j] = 1
                domination_count[j] += 1
            elif j_dom_i and j_any_better:
                dom_matrix[j, i] = 1
                domination_count[i] += 1

    # Peel off fronts
    current_rank = 0
    assigned = 0
    while assigned < n:
        found_any = False
        for i in range(n):
            if ranks[i] == -1 and domination_count[i] == 0:
                ranks[i] = current_rank
                assigned += 1
                found_any = True

        for i in range(n):
            if ranks[i] == current_rank:
                for j in range(n):
                    if dom_matrix[i, j] == 1 and ranks[j] == -1:
                        domination_count[j] -= 1

        current_rank += 1
        if not found_any:
            for i in range(n):
                if ranks[i] == -1:
                    ranks[i] = current_rank
                    assigned += 1
            break

    return ranks


def sbx_crossover_cy(np.ndarray[np.float64_t, ndim=1] parent1,
                      np.ndarray[np.float64_t, ndim=1] parent2,
                      double eta_c):
    """
    Simulated Binary Crossover compiled to C.

    Args:
        parent1, parent2: (N,) float arrays.
        eta_c: distribution index.

    Returns:
        child1, child2: (N,) float arrays.
    """
    cdef int n = parent1.shape[0]
    cdef np.ndarray[np.float64_t, ndim=1] child1 = np.empty(n, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] child2 = np.empty(n, dtype=np.float64)
    cdef double u, beta, midpoint, spread
    cdef int i

    for i in range(n):
        u = _rand_uniform()
        if u <= 0.5:
            beta = pow(2.0 * u, 1.0 / (eta_c + 1.0))
        else:
            beta = pow(2.0 * (1.0 - u), -1.0 / (eta_c + 1.0))
        midpoint = (parent1[i] + parent2[i]) * 0.5
        spread = fabs(parent1[i] - parent2[i]) * 0.5
        child1[i] = midpoint + beta * spread
        child2[i] = midpoint - beta * spread

    return child1, child2


def gaussian_mutate_cy(np.ndarray[np.float64_t, ndim=1] features,
                        double mutation_prob,
                        double mutation_sigma):
    """
    Gaussian mutation compiled to C.

    Args:
        features: (N,) float array (modified in-place).
        mutation_prob: per-gene mutation probability.
        mutation_sigma: Gaussian noise std dev.

    Returns:
        features: mutated array.
    """
    cdef int n = features.shape[0]
    cdef int i

    for i in range(n):
        if _rand_uniform() < mutation_prob:
            features[i] += _rand_normal(0.0, mutation_sigma)
            if features[i] < 0.0:
                features[i] = 0.0
            elif features[i] > 1.0:
                features[i] = 1.0

    return features


def seed_rng(unsigned int seed):
    """Seed the C random number generator."""
    srand(seed)
