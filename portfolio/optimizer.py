"""
Modified NSGA-II for portfolio optimization (Lou 2023).

Implements three modifications from:
    Lou, "Optimizing Portfolios with Modified NSGA-II Solutions",
    IEEE ICDSCA 2023, pp. 375-380.

1. Refined Selection (Section III-A):
   Bias tournament candidate selection toward less crowded solutions.
   P(i) = c(i)^b / sum(c(j)^b), where b is a bias parameter.

2. Dynamic Mutation (Section III-B):
   As the Pareto front gets denser, increase mutation probability
   and sigma to promote exploration. Decay both over generations.

3. Optimized Initialization (Section III-C):
   Start with a large population for one generation, then select
   down to standard size using non-dominated sorting + crowding distance.
"""

import random
import math
import numpy as np
from nsga2.utils import NSGA2Utils
from nsga2.population import Population
from nsga2.individual import Individual
from tqdm import tqdm
from portfolio.cython_ops import (
    fast_nondominated_sort_cy,
    sbx_crossover_cy,
    gaussian_mutate_cy,
)


class PortfolioNSGA2Utils(NSGA2Utils):
    """NSGA-II utilities with Lou 2023 modifications for portfolio optimization."""

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9,
                 crossover_param=2, mutation_param=5,
                 selection_bias=0.5, init_population_multiplier=10,
                 base_mutation_prob=0.3, base_mutation_sigma=0.1,
                 mutation_decay=0.995,
                 use_lou_selection=True, use_lou_init=True,
                 use_lou_mutation=True):
        super().__init__(problem, num_of_individuals, num_of_tour_particips,
                         tournament_prob, crossover_param, mutation_param)

        # Lou Modification 1: Refined selection
        self.use_lou_selection = use_lou_selection
        self.selection_bias = selection_bias  # b parameter (0=standard, 1=strict crowding)

        # Lou Modification 2: Dynamic mutation
        self.use_lou_mutation = use_lou_mutation
        self.base_mutation_prob = base_mutation_prob
        self.base_mutation_sigma = base_mutation_sigma
        self.mutation_prob = base_mutation_prob
        self.mutation_sigma = base_mutation_sigma
        self.mutation_decay = mutation_decay  # per-generation decay

        # Lou Modification 3: Optimized initialization
        self.use_lou_init = use_lou_init
        self.init_population_multiplier = init_population_multiplier

    # -- Cython-accelerated non-dominated sorting --

    def fast_nondominated_sort(self, population):
        """Non-dominated sorting using Cython-compiled inner loops."""
        individuals = population.population
        n = len(individuals)
        if n == 0:
            population.fronts = [[]]
            return

        m = len(individuals[0].objectives)
        objectives = np.empty((n, m), dtype=np.float64)
        for i, ind in enumerate(individuals):
            for j in range(m):
                objectives[i, j] = ind.objectives[j]

        ranks = fast_nondominated_sort_cy(objectives)

        max_rank = int(ranks.max())
        population.fronts = [[] for _ in range(max_rank + 2)]
        for i, ind in enumerate(individuals):
            ind.rank = int(ranks[i])
            ind.domination_count = 0
            ind.dominated_solutions = []
            population.fronts[ind.rank].append(ind)
        population.fronts.append([])

    # -- Lou Modification 3: Optimized Initialization (Section III-C) --
    # Start with large population, run 1 generation of sorting,
    # then select down to standard size.

    def create_initial_population(self):
        """
        Generate a large initial population, evaluate and sort it,
        then select the best N individuals by rank + crowding distance.
        """
        if not self.use_lou_init:
            return super().create_initial_population()

        large_size = self.num_of_individuals * self.init_population_multiplier
        large_pop = Population()

        for _ in range(large_size):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            large_pop.append(individual)

        # Sort and compute crowding distances on the large population
        self.fast_nondominated_sort(large_pop)
        for front in large_pop.fronts:
            self.calculate_crowding_distance(front)

        # Select best N individuals by front rank, then crowding distance
        selected = Population()
        front_num = 0
        while (front_num < len(large_pop.fronts)
               and len(large_pop.fronts[front_num]) > 0
               and len(selected) + len(large_pop.fronts[front_num]) <= self.num_of_individuals):
            selected.extend(large_pop.fronts[front_num])
            front_num += 1

        # Fill remaining from the next partial front
        if len(selected) < self.num_of_individuals and front_num < len(large_pop.fronts):
            remaining_front = large_pop.fronts[front_num]
            remaining_front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            remaining = self.num_of_individuals - len(selected)
            selected.extend(remaining_front[:remaining])

        return selected

    # -- Lou Modification 2: Dynamic Mutation (Section III-B) --
    # As the Pareto front gets denser, increase mutation probability and sigma.
    # Apply decay over generations so convergence still happens.

    def compute_front_density(self, population):
        """
        Compute density of the first Pareto front.
        Density = number of front-0 individuals / total population.
        Higher density means more solutions are converging to the front.
        """
        if not population.fronts or len(population.fronts[0]) == 0:
            return 0.0
        return len(population.fronts[0]) / len(population)

    def update_dynamic_mutation(self, generation, front_density):
        """
        Adjust mutation probability and sigma based on front density.
        Denser front -> higher mutation prob and sigma (explore more).
        Apply decay over generations to allow eventual convergence.
        """
        if not self.use_lou_mutation:
            return

        # Scale mutation by front density (denser = more mutation)
        density_factor = 1.0 + front_density

        # Apply generation decay
        decay = self.mutation_decay ** generation

        self.mutation_prob = self.base_mutation_prob * density_factor * decay
        self.mutation_sigma = self.base_mutation_sigma * density_factor * decay

        # Clamp to reasonable bounds
        self.mutation_prob = min(1.0, max(0.01, self.mutation_prob))
        self.mutation_sigma = min(0.5, max(0.001, self.mutation_sigma))

    # -- Lou Modification 1: Refined Selection (Section III-A) --
    # P(i) = c(i)^b / sum(c(j)^b)
    # Bias tournament candidate selection toward less crowded solutions.

    def _tournament(self, population):
        """
        Tournament selection with crowding-distance-biased candidate sampling.
        Instead of uniform random selection of candidates, probability of
        being selected is proportional to crowding_distance^b.
        """
        if not self.use_lou_selection or self.selection_bias == 0:
            # Standard uniform random tournament
            participants = random.sample(population.population, self.num_of_tour_particips)
        else:
            # Lou's biased selection: P(i) = c(i)^b / sum(c(j)^b)
            participants = self._biased_sample(population, self.num_of_tour_particips)

        # Tournament comparison: rank first, then crowding distance
        best = None
        for participant in participants:
            if best is None or self._tournament_compare(participant, best) == 1:
                best = participant
        return best

    def _biased_sample(self, population, k):
        """Sample k individuals with probability proportional to crowding_distance^b."""
        individuals = population.population
        b = self.selection_bias

        # Compute selection probabilities
        weights = []
        for ind in individuals:
            cd = ind.crowding_distance if ind.crowding_distance is not None else 0.0
            # Cap infinity values for numerical stability
            cd = min(cd, 1e6)
            weights.append(cd ** b)

        total = sum(weights)
        if total == 0:
            return random.sample(individuals, k)

        probs = [w / total for w in weights]

        # Weighted sampling without replacement
        selected = []
        remaining_indices = list(range(len(individuals)))
        remaining_probs = list(probs)

        for _ in range(min(k, len(individuals))):
            prob_sum = sum(remaining_probs)
            if prob_sum == 0:
                idx_in_remaining = random.randint(0, len(remaining_indices) - 1)
            else:
                r = random.random() * prob_sum
                cumulative = 0.0
                idx_in_remaining = 0
                for j, p in enumerate(remaining_probs):
                    cumulative += p
                    if cumulative >= r:
                        idx_in_remaining = j
                        break

            selected.append(individuals[remaining_indices[idx_in_remaining]])
            remaining_indices.pop(idx_in_remaining)
            remaining_probs.pop(idx_in_remaining)

        return selected

    def _tournament_compare(self, ind1, ind2):
        """
        Lou's tournament comparison (from paper's T(i,j) formula):
        Prefer lower rank, or same rank with higher crowding distance.
        """
        if ind1.rank < ind2.rank:
            return 1
        if ind1.rank > ind2.rank:
            return -1
        if (ind1.crowding_distance or 0) > (ind2.crowding_distance or 0):
            return 1
        if (ind1.crowding_distance or 0) < (ind2.crowding_distance or 0):
            return -1
        return 1

    # -- Portfolio-adapted genetic operators --
    # Base NSGA2Utils uses __double_underscore (name-mangled) methods,
    # so we reimplement create_children and its helpers.

    def create_children(self, population):
        """Create offspring with Lou's dynamic mutation and portfolio constraints."""
        children = []
        while len(children) < len(population):
            parent1 = self._tournament(population)
            parent2 = parent1
            attempts = 0
            while parent1 == parent2 and attempts < 10:
                parent2 = self._tournament(population)
                attempts += 1
            child1, child2 = self._crossover(parent1, parent2)
            self._mutate(child1)
            self._mutate(child2)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)
        return children

    def _crossover(self, parent1, parent2):
        """SBX crossover using Cython."""
        p1 = np.array(parent1.features, dtype=np.float64)
        p2 = np.array(parent2.features, dtype=np.float64)
        c1_arr, c2_arr = sbx_crossover_cy(p1, p2, float(self.crossover_param))
        child1 = Individual()
        child2 = Individual()
        child1.features = c1_arr.tolist()
        child2.features = c2_arr.tolist()
        return child1, child2

    def _mutate(self, child):
        """Gaussian mutation using Cython."""
        features = np.array(child.features, dtype=np.float64)
        gaussian_mutate_cy(features, self.mutation_prob, self.mutation_sigma)
        child.features = features.tolist()


class PortfolioEvolution:
    """
    Modified NSGA-II evolution with Lou 2023 modifications.

    Lou's paper tests three configurations:
    1. Standard NSGA-II (baseline)
    2. NSGA-II + refined selection
    3. NSGA-II + refined selection + dynamic mutation + optimized init (fully optimized)
    """

    def __init__(self, problem, num_of_generations=200, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9,
                 crossover_param=2, mutation_param=5,
                 selection_bias=0.5, init_population_multiplier=10,
                 base_mutation_prob=0.3, base_mutation_sigma=0.1,
                 mutation_decay=0.995,
                 use_lou_selection=True, use_lou_init=True,
                 use_lou_mutation=True):
        self.utils = PortfolioNSGA2Utils(
            problem, num_of_individuals, num_of_tour_particips,
            tournament_prob, crossover_param, mutation_param,
            selection_bias=selection_bias,
            init_population_multiplier=init_population_multiplier,
            base_mutation_prob=base_mutation_prob,
            base_mutation_sigma=base_mutation_sigma,
            mutation_decay=mutation_decay,
            use_lou_selection=use_lou_selection,
            use_lou_init=use_lou_init,
            use_lou_mutation=use_lou_mutation,
        )
        self.population = None
        self.num_of_generations = num_of_generations
        self.num_of_individuals = num_of_individuals

    def evolve(self):
        """Run the modified NSGA-II evolution loop. Returns Pareto front."""
        # Optimized initialization (Lou Mod 3)
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)

        returned_population = None
        for gen in tqdm(range(self.num_of_generations), desc="NSGA-II"):
            # Merge parents + children (2N)
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)

            # Environmental selection: fill new population by front rank
            new_population = Population()
            front_num = 0
            while (len(new_population) + len(self.population.fronts[front_num])
                   <= self.num_of_individuals):
                self.utils.calculate_crowding_distance(self.population.fronts[front_num])
                new_population.extend(self.population.fronts[front_num])
                front_num += 1

            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(
                key=lambda ind: ind.crowding_distance, reverse=True
            )
            remaining = self.num_of_individuals - len(new_population)
            new_population.extend(self.population.fronts[front_num][:remaining])

            returned_population = self.population
            self.population = new_population

            # Dynamic mutation update (Lou Mod 2)
            self.utils.fast_nondominated_sort(self.population)
            front_density = self.utils.compute_front_density(self.population)
            self.utils.update_dynamic_mutation(gen, front_density)

            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)

        return returned_population.fronts[0]
