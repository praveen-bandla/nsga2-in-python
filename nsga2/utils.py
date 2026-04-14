import numpy as np
import numba as nb
import random
from nsga2.population import Population


@nb.jit(nopython=True)
def _compute_domination(costs):
    """
    Compute non-dominated ranks for a population.

    Parameters
    ----------
    costs : np.ndarray, shape (N, K)
        Objective values for N individuals across K objectives.
        All objectives are assumed to be minimized.

    Returns
    -------
    ranks : np.ndarray, shape (N,), dtype int32
        Front rank of each individual (0 = Pareto front).
    """
    n = costs.shape[0]
    # how many dominate i
    domination_count = np.zeros(n, dtype=np.int32)
    # dominated_by[i] = list of j that i dominates
    dominated_by = np.zeros((n, n), dtype=np.int32)
    # number of solutions i dominates
    dominated_by_count = np.zeros(n, dtype=np.int32) 
    for i in range(n):
        for j in range(i + 1, n):
            i_dom_j = True
            j_dom_i = True
            i_strictly_better = False
            j_strictly_better = False

            for k in range(costs.shape[1]):
                if costs[i, k] > costs[j, k]:
                    i_dom_j = False
                if costs[j, k] > costs[i, k]:
                    j_dom_i = False
                if costs[i, k] < costs[j, k]:
                    i_strictly_better = True
                if costs[j, k] < costs[i, k]:
                    j_strictly_better = True

            if i_dom_j and i_strictly_better:
                dominated_by[i, dominated_by_count[i]] = j
                dominated_by_count[i] += 1
                domination_count[j] += 1
            elif j_dom_i and j_strictly_better:
                dominated_by[j, dominated_by_count[j]] = i
                dominated_by_count[j] += 1
                domination_count[i] += 1

    ranks = np.full(n, -1, dtype=np.int32)
    current_front = np.where(domination_count == 0)[0]
    rank = 0

    while len(current_front) > 0:
        next_front = []
        for i in current_front:
            ranks[i] = rank
            for k in range(dominated_by_count[i]):
                j = dominated_by[i, k]
                domination_count[j] -= 1
                if domination_count[j] == 0:
                    next_front.append(j)
        rank += 1
        current_front = np.array(next_front, dtype=np.int64)

    return ranks


@nb.jit(nopython=True)
def _crowding_distance(costs):
    """
    Compute crowding distance for a set of solutions within a single front.

    Parameters
    ----------
    costs : np.ndarray, shape (M, K)
        Objective values for M solutions across K objectives.

    Returns
    -------
    distances : np.ndarray, shape (M,), dtype float64
        Crowding distance for each solution.
    """
    m = costs.shape[0]
    k = costs.shape[1]
    distances = np.zeros(m, dtype=np.float64)

    if m <= 2:
        for i in range(m):
            distances[i] = np.inf
        return distances

    for obj in range(k):
        col = costs[:, obj]
        sort_ids = np.argsort(col)

        distances[sort_ids[0]] = np.inf
        distances[sort_ids[m - 1]] = np.inf

        obj_min = col[sort_ids[0]]
        obj_max = col[sort_ids[m - 1]]
        scale = obj_max - obj_min
        if scale == 0.0:
            scale = 1.0

        for i in range(1, m - 1):
            distances[sort_ids[i]] += (col[sort_ids[i + 1]] - col[sort_ids[i - 1]]) / scale

    return distances


@nb.jit(nopython=True)
def _sbx_crossover(f1, f2, eta):
    """
    Simulated Binary Crossover (SBX) on two feature vectors.

    Parameters
    ----------
    f1, f2 : np.ndarray, shape (D,)
        Parent feature vectors.
    eta : float
        Distribution index (crossover_param).

    Returns
    -------
    c1, c2 : np.ndarray, shape (D,)
        Offspring feature vectors.
    """
    n = f1.shape[0]
    c1 = np.empty(n, dtype=np.float64)
    c2 = np.empty(n, dtype=np.float64)

    for i in range(n):
        u = np.random.random()
        if u <= 0.5:
            beta = (2.0 * u) ** (1.0 / (eta + 1.0))
        else:
            beta = (2.0 * (1.0 - u)) ** (-1.0 / (eta + 1.0))

        mid = (f1[i] + f2[i]) / 2.0
        spread = abs(f1[i] - f2[i]) / 2.0
        c1[i] = mid + beta * spread
        c2[i] = mid - beta * spread

    return c1, c2


@nb.jit(nopython=True)
def _polynomial_mutation(features, lower, upper, eta):
    """
    Polynomial mutation on a feature vector.

    Parameters
    ----------
    features : np.ndarray, shape (D,)
        Individual's feature vector.
    lower, upper : np.ndarray, shape (D,)
        Per-gene lower and upper bounds.
    eta : float
        Distribution index (mutation_param).

    Returns
    -------
    mutated : np.ndarray, shape (D,)
        Mutated feature vector (clipped to bounds).
    """
    n = features.shape[0]
    mutated = features.copy()

    for i in range(n):
        u = np.random.random()
        if u < 0.5:
            delta = (2.0 * u) ** (1.0 / (eta + 1.0)) - 1.0
            mutated[i] += delta * (features[i] - lower[i])
        else:
            delta = 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (eta + 1.0))
            mutated[i] += delta * (upper[i] - features[i])

        mutated[i] = max(lower[i], min(upper[i], mutated[i]))

    return mutated


# ---------------------------------------------------------------------------
# NSGA2Utils — class methods are thin wrappers around the JIT kernels
# ---------------------------------------------------------------------------

class NSGA2Utils:

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9,
                 crossover_param=2, mutation_param=5):

        self.problem = problem
        self.num_of_individuals = num_of_individuals
        self.num_of_tour_particips = num_of_tour_particips
        self.tournament_prob = tournament_prob
        self.crossover_param = crossover_param
        self.mutation_param = mutation_param

        # Cache bounds arrays for mutation kernel
        self._lower = np.array([r[0] for r in problem.variables_range], dtype=np.float64)
        self._upper = np.array([r[1] for r in problem.variables_range], dtype=np.float64)

    def create_initial_population(self):
        population = Population()
        for _ in range(self.num_of_individuals):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)
        return population

    def fast_nondominated_sort(self, population):
        """
        Assign front ranks to all individuals using the JIT-compiled dominance kernel.
        Writes individual.rank and rebuilds population.fronts.
        """
        individuals = list(population)

        # Extract objectives into a (N, K) array for the JIT kernel
        costs = np.array([ind.objectives for ind in individuals], dtype=np.float64)

        ranks = _compute_domination(costs)

        # Write ranks back and rebuild fronts
        max_rank = int(ranks.max())
        population.fronts = [[] for _ in range(max_rank + 1)]
        for ind, r in zip(individuals, ranks):
            ind.rank = int(r)
            population.fronts[r].append(ind)

    def calculate_crowding_distance(self, front):
        """
        Assign crowding distances to individuals in a single front using the JIT kernel.
        """
        if len(front) == 0:
            return

        # Extract objectives into (M, K) array
        costs = np.array([ind.objectives for ind in front], dtype=np.float64)

        distances = _crowding_distance(costs)

        for ind, d in zip(front, distances):
            ind.crowding_distance = float(d)

    def crowding_operator(self, individual, other_individual):
        if (individual.rank < other_individual.rank) or \
                ((individual.rank == other_individual.rank) and (
                        individual.crowding_distance > other_individual.crowding_distance)):
            return 1
        else:
            return -1

    def create_children(self, population):
        children = []
        while len(children) < len(population):
            parent1 = self.__tournament(population)
            parent2 = parent1
            while parent1 == parent2:
                parent2 = self.__tournament(population)
            child1, child2 = self.__crossover(parent1, parent2)
            self.__mutate(child1)
            self.__mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)
        return children

    def __crossover(self, individual1, individual2):
        child1 = self.problem.generate_individual()
        child2 = self.problem.generate_individual()

        f1 = np.array(individual1.features, dtype=np.float64)
        f2 = np.array(individual2.features, dtype=np.float64)

        c1, c2 = _sbx_crossover(f1, f2, float(self.crossover_param))

        child1.features = c1
        child2.features = c2
        return child1, child2

    def __mutate(self, child):
        features = np.asarray(child.features, dtype=np.float64)
        child.features = _polynomial_mutation(
            features, self._lower, self._upper, float(self.mutation_param)
        )

    def __tournament(self, population):
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1 and self.__choose_with_prob(self.tournament_prob)):
                best = participant
        return best

    def __choose_with_prob(self, prob):
        if random.random() <= prob:
            return True
        return False
