# pyright: reportMissingImports=false

import random
from importlib import import_module

import numpy as np
from tqdm import tqdm

from nsga2.individual import Individual
from nsga2.population import Population
from nsga2.utils import NSGA2Utils
from portfolio.numba_ops import gaussian_mutate_jit, sbx_crossover_jit

try:
    fast_nondominated_sort_cy = import_module(
        "portfolio.cython_ops"
    ).fast_nondominated_sort_cy
except ModuleNotFoundError:  # pragma: no cover
    fast_nondominated_sort_cy = None


class PortfolioNSGA2Utils(NSGA2Utils):
    def __init__(
        self,
        problem,
        num_of_individuals=100,
        num_of_tour_particips=2,
        tournament_prob=0.9,
        crossover_param=2,
        mutation_param=5,
        selection_bias=0.5,
        init_population_multiplier=10,
        base_mutation_prob=0.3,
        base_mutation_sigma=0.1,
        mutation_decay=0.995,
        use_lou_selection=True,
        use_lou_init=True,
        use_lou_mutation=True,
    ):
        super().__init__(
            problem,
            num_of_individuals,
            num_of_tour_particips,
            tournament_prob,
            crossover_param,
            mutation_param,
        )
        self.use_lou_selection = use_lou_selection
        self.selection_bias = selection_bias
        self.use_lou_mutation = use_lou_mutation
        self.base_mutation_prob = base_mutation_prob
        self.base_mutation_sigma = base_mutation_sigma
        self.mutation_prob = base_mutation_prob
        self.mutation_sigma = base_mutation_sigma
        self.mutation_decay = mutation_decay
        self.use_lou_init = use_lou_init
        self.init_population_multiplier = init_population_multiplier

    def fast_nondominated_sort(self, population):
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

        if fast_nondominated_sort_cy is None:
            return super().fast_nondominated_sort(population)

        ranks = fast_nondominated_sort_cy(objectives)
        max_rank = int(ranks.max())
        population.fronts = [[] for _ in range(max_rank + 2)]

        for i, ind in enumerate(individuals):
            ind.rank = int(ranks[i])
            ind.domination_count = 0
            ind.dominated_solutions = []
            population.fronts[ind.rank].append(ind)

        population.fronts.append([])

    def create_initial_population(self):
        if not self.use_lou_init:
            return super().create_initial_population()

        large_size = self.num_of_individuals * self.init_population_multiplier
        large_pop = Population()

        for _ in range(large_size):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            large_pop.append(individual)

        self.fast_nondominated_sort(large_pop)
        for front in large_pop.fronts:
            self.calculate_crowding_distance(front)

        selected = Population()
        front_num = 0
        while (
            front_num < len(large_pop.fronts)
            and len(large_pop.fronts[front_num]) > 0
            and len(selected) + len(large_pop.fronts[front_num])
            <= self.num_of_individuals
        ):
            selected.extend(large_pop.fronts[front_num])
            front_num += 1

        if len(selected) < self.num_of_individuals and front_num < len(
            large_pop.fronts
        ):
            remaining_front = large_pop.fronts[front_num]
            remaining_front.sort(key=lambda ind: ind.crowding_distance, reverse=True)
            remaining = self.num_of_individuals - len(selected)
            selected.extend(remaining_front[:remaining])

        return selected

    def compute_front_density(self, population):
        if not population.fronts or len(population.fronts[0]) == 0:
            return 0.0
        return len(population.fronts[0]) / len(population)

    def update_dynamic_mutation(self, generation, front_density):
        if not self.use_lou_mutation:
            return

        density_factor = 1.0 + front_density
        decay = self.mutation_decay**generation
        self.mutation_prob = self.base_mutation_prob * density_factor * decay
        self.mutation_sigma = self.base_mutation_sigma * density_factor * decay
        self.mutation_prob = min(1.0, max(0.01, self.mutation_prob))
        self.mutation_sigma = min(0.5, max(0.001, self.mutation_sigma))

    def _tournament(self, population):
        if not self.use_lou_selection or self.selection_bias == 0:
            participants = random.sample(
                population.population, self.num_of_tour_particips
            )
        else:
            participants = self._biased_sample(population, self.num_of_tour_particips)

        best = None
        for participant in participants:
            if best is None or self._tournament_compare(participant, best) == 1:
                best = participant
        return best

    def _biased_sample(self, population, k):
        individuals = population.population
        b = self.selection_bias
        weights = []

        for ind in individuals:
            cd = ind.crowding_distance if ind.crowding_distance is not None else 0.0
            cd = min(cd, 1e6)
            weights.append(cd**b)

        total = sum(weights)
        if total == 0:
            return random.sample(individuals, k)

        probs = [w / total for w in weights]
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
        if ind1.rank < ind2.rank:
            return 1
        if ind1.rank > ind2.rank:
            return -1
        if (ind1.crowding_distance or 0) > (ind2.crowding_distance or 0):
            return 1
        if (ind1.crowding_distance or 0) < (ind2.crowding_distance or 0):
            return -1
        return 1

    def create_children(self, population):
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
        p1 = np.array(parent1.features, dtype=np.float64)
        p2 = np.array(parent2.features, dtype=np.float64)
        c1_arr, c2_arr = sbx_crossover_jit(p1, p2, float(self.crossover_param))
        child1 = Individual()
        child2 = Individual()
        child1.features = c1_arr
        child2.features = c2_arr
        return child1, child2

    def _mutate(self, child):
        features = np.array(child.features, dtype=np.float64)
        gaussian_mutate_jit(features, self.mutation_prob, self.mutation_sigma)
        child.features = features


class PortfolioEvolution:
    def __init__(
        self,
        problem,
        num_of_generations=200,
        num_of_individuals=100,
        num_of_tour_particips=2,
        tournament_prob=0.9,
        crossover_param=2,
        mutation_param=5,
        selection_bias=0.5,
        init_population_multiplier=10,
        base_mutation_prob=0.3,
        base_mutation_sigma=0.1,
        mutation_decay=0.995,
        use_lou_selection=True,
        use_lou_init=True,
        use_lou_mutation=True,
    ):
        self.utils = PortfolioNSGA2Utils(
            problem,
            num_of_individuals,
            num_of_tour_particips,
            tournament_prob,
            crossover_param,
            mutation_param,
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
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)

        returned_population = self.population
        for gen in tqdm(range(self.num_of_generations), desc="NSGA-II"):
            self.population.extend(children)
            self.utils.fast_nondominated_sort(self.population)

            new_population = Population()
            front_num = 0
            while (
                len(new_population) + len(self.population.fronts[front_num])
                <= self.num_of_individuals
            ):
                self.utils.calculate_crowding_distance(
                    self.population.fronts[front_num]
                )
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

            self.utils.fast_nondominated_sort(self.population)
            front_density = self.utils.compute_front_density(self.population)
            self.utils.update_dynamic_mutation(gen, front_density)

            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)

        assert returned_population is not None
        return returned_population.fronts[0]
