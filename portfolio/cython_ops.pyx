# cython: boundscheck=False, wraparound=False, cdivision=True
"""
Cython-compiled non-dominated sorting for NSGA-II.
AOT-compiled to C for zero runtime overhead.
"""

import numpy as np
cimport numpy as np

np.import_array()


def fast_nondominated_sort_cy(np.ndarray[np.float64_t, ndim=2] objectives):
    """
    Non-dominated sorting compiled to C.

    Args:
        objectives: (N, M) array, all objectives minimized.

    Returns:
        ranks: (N,) int array with Pareto front rank.
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
