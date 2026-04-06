"""
Modified NSGA-II optimizer for portfolio optimization (Lou 2023).

Extends the base NSGA-II with three modifications from:
    Lou (2023), "Multi-Objective Portfolio Optimization with Modified NSGA-II",
    IEEE ICDSCA 2023.

Modifications:
    1. Smart Initialization - mix of equal-weight, concentrated, sparse,
       and random portfolios instead of purely random.
    2. Adaptive Mutation - dynamically adjust mutation strength based on
       population diversity to prevent premature convergence.
    3. Refined Selection - enhanced tournament selection with objective
       spread as an additional tie-breaking criterion.

The base PortfolioNSGA2Utils also reimplements genetic operators to handle
portfolio weight constraints (non-negative, sum to 1) since the parent
class uses name-mangled private methods that cannot be overridden.
"""

import random
import math
from nsga2.utils import NSGA2Utils
from nsga2.population import Population
from nsga2.individual import Individual
from tqdm import tqdm


class PortfolioNSGA2Utils(NSGA2Utils):
    """
    NSGA-II utilities adapted for portfolio optimization with Lou 2023 mods.

    Args:
        problem: PortfolioProblem instance.
        num_of_individuals: Population size.
        num_of_tour_particips: Tournament size.
        tournament_prob: Probability of selecting the better individual.
        crossover_param: SBX distribution index (eta_c).
        mutation_param: Polynomial mutation distribution index (eta_m).
        adaptive_mutation: Enable Lou 2023 adaptive mutation.
        smart_init: Enable Lou 2023 smart initialization.
        refined_selection: Enable Lou 2023 refined selection.
    """

    def __init__(self, problem, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9,
                 crossover_param=2, mutation_param=5,
                 adaptive_mutation=True, smart_init=True,
                 refined_selection=True):
        super().__init__(problem, num_of_individuals, num_of_tour_particips,
                         tournament_prob, crossover_param, mutation_param)
        self.adaptive_mutation = adaptive_mutation
        self.smart_init = smart_init
        self.refined_selection = refined_selection

        # Adaptive mutation state
        self.base_mutation_param = mutation_param
        self.min_mutation_param = 1
        self.max_mutation_param = 20

    # Lou 2023 Modification 1: Smart Initialization
    def create_initial_population(self):
        """
        Create initial population with diverse strategies (Lou 2023).

        Instead of purely random portfolios, uses a mix:
            - 10% equal-weight (with small perturbation)
            - 20% concentrated (heavy weight in 1 stock)
            - 20% sparse (5-15 stocks selected)
            - 50% random

        Falls back to standard random initialization if smart_init=False.
        """
        if not self.smart_init:
            return super().create_initial_population()

        population = Population()
        n = self.num_of_individuals
        num_assets = self.problem.num_assets

        # Equal-weight portfolios (10%)
        n_equal = max(1, n // 10)
        for _ in range(n_equal):
            individual = Individual()
            base = 1.0 / num_assets
            features = [base + random.gauss(0, base * 0.1) for _ in range(num_assets)]
            features = [max(0.0, f) for f in features]
            total = sum(features)
            individual.features = [f / total for f in features]
            self.problem.calculate_objectives(individual)
            population.append(individual)

        # Concentrated portfolios (20%)
        n_concentrated = max(1, n // 5)
        for _ in range(n_concentrated):
            individual = Individual()
            features = [0.0] * num_assets
            # Pick 1-3 focus stocks with 30-60% total weight
            n_focus = random.randint(1, 3)
            focus_idxs = random.sample(range(num_assets), n_focus)
            focus_weight = random.uniform(0.3, 0.6)
            for idx in focus_idxs:
                features[idx] = focus_weight / n_focus
            # Distribute remaining weight randomly
            remaining = 1.0 - focus_weight
            for i in range(num_assets):
                if i not in focus_idxs:
                    features[i] = random.random()
            # Normalize non-focus to fill remaining weight
            non_focus_total = sum(features[i] for i in range(num_assets) if i not in focus_idxs)
            if non_focus_total > 0:
                for i in range(num_assets):
                    if i not in focus_idxs:
                        features[i] = features[i] / non_focus_total * remaining
            individual.features = features
            self.problem.calculate_objectives(individual)
            population.append(individual)

        # Sparse portfolios (20%)
        n_sparse = max(1, n // 5)
        for _ in range(n_sparse):
            individual = Individual()
            k = random.randint(5, min(15, num_assets))
            selected = random.sample(range(num_assets), k)
            features = [0.0] * num_assets
            for idx in selected:
                features[idx] = random.random()
            total = sum(features)
            individual.features = [f / total for f in features]
            self.problem.calculate_objectives(individual)
            population.append(individual)

        # Random portfolios (remaining)
        n_random = n - n_equal - n_concentrated - n_sparse
        for _ in range(n_random):
            individual = self.problem.generate_individual()
            self.problem.calculate_objectives(individual)
            population.append(individual)

        return population

    # Lou 2023 Modification 2: Adaptive Mutation

    def compute_population_diversity(self, population):
        """
        Compute diversity as average pairwise Euclidean distance in
        objective space. Higher = more diverse population.

        This is O(N^2) in population size - intentionally pure Python
        for the baseline (will be optimized later with NumPy/Numba).
        """
        individuals = list(population)
        n = len(individuals)
        if n <= 1:
            return 0.0

        total_dist = 0.0
        count = 0
        for i in range(n):
            for j in range(i + 1, n):
                dist_sq = 0.0
                for a, b in zip(individuals[i].objectives, individuals[j].objectives):
                    dist_sq += (a - b) ** 2
                total_dist += math.sqrt(dist_sq)
                count += 1

        return total_dist / count

    def adapt_mutation_param(self, diversity, initial_diversity):
        """
        Adjust mutation distribution index based on population diversity.

        Low diversity  -> decrease eta_m -> stronger mutation (explore)
        High diversity -> increase eta_m -> weaker mutation (exploit)

        The mutation_param (eta_m) controls the spread of polynomial
        mutation: lower eta_m = more spread = more exploration.
        """
        if not self.adaptive_mutation or initial_diversity < 1e-12:
            return self.mutation_param

        ratio = diversity / initial_diversity

        if ratio < 0.3:
            # case when converges too fast so increase mutation strength
            self.mutation_param = max(
                self.min_mutation_param,
                self.base_mutation_param * ratio
            )
        elif ratio > 0.7:
            # good spread, weaker mutation
            self.mutation_param = min(
                self.max_mutation_param,
                self.base_mutation_param / max(ratio, 0.01)
            )
        else:
            # else base param
            self.mutation_param = self.base_mutation_param

        return self.mutation_param

    # Lou 2023 Modification 3: Refined Selection
    def crowding_operator(self, individual, other_individual):
        """
        Enhanced crowding comparison operator (Lou 2023).

        Standard NSGA-II: compare by rank, then crowding distance.
        Lou 2023 adds: when rank and crowding distance tie, prefer the
        solution with greater spread across objectives (better tradeoff
        representation).
        """
        if not self.refined_selection:
            return super().crowding_operator(individual, other_individual)

        # lower rank better
        if individual.rank < other_individual.rank:
            return 1
        if individual.rank > other_individual.rank:
            return -1

        # crowding distance higher better
        if individual.crowding_distance > other_individual.crowding_distance:
            return 1
        if individual.crowding_distance < other_individual.crowding_distance:
            return -1

        # objective spread higher better
        if individual.objectives and other_individual.objectives:
            spread1 = max(individual.objectives) - min(individual.objectives)
            spread2 = max(other_individual.objectives) - min(other_individual.objectives)
            if spread1 > spread2:
                return 1
            if spread1 < spread2:
                return -1

        return 1

    # Portfolio-adapted genetic operators
    # The base NSGA2Utils uses __double_underscore (name-mangled) methods
    # for crossover, mutation, and tournament, so we must reimplement
    # create_children entirely rather than just overriding the operators.

    def create_children(self, population):
        """Create offspring with portfolio constraint enforcement."""
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
            # calculate_objectives includes weight repair (sum to 1)
            self.problem.calculate_objectives(child1)
            self.problem.calculate_objectives(child2)
            children.append(child1)
            children.append(child2)
        return children

    def _tournament(self, population):
        """Binary tournament selection."""
        participants = random.sample(population.population, self.num_of_tour_particips)
        best = None
        for participant in participants:
            if best is None or (
                    self.crowding_operator(participant, best) == 1
                    and random.random() <= self.tournament_prob):
                best = participant
        return best

    def _crossover(self, parent1, parent2):
        """Simulated Binary Crossover (SBX) for portfolio weights."""
        child1 = Individual()
        child2 = Individual()
        n = self.problem.num_assets
        child1.features = [0.0] * n
        child2.features = [0.0] * n

        for i in range(n):
            beta = self._get_beta()
            midpoint = (parent1.features[i] + parent2.features[i]) / 2
            spread = abs(parent1.features[i] - parent2.features[i]) / 2
            child1.features[i] = midpoint + beta * spread
            child2.features[i] = midpoint - beta * spread

        return child1, child2

    def _get_beta(self):
        """Generate beta parameter for SBX crossover."""
        u = random.random()
        eta = self.crossover_param
        if u <= 0.5:
            return (2 * u) ** (1 / (eta + 1))
        return (2 * (1 - u)) ** (-1 / (eta + 1))

    def _mutate(self, child):
        """Polynomial mutation for portfolio weights."""
        n = self.problem.num_assets
        for gene in range(n):
            u = random.random()
            eta = self.mutation_param

            if u < 0.5:
                delta = (2 * u) ** (1 / (eta + 1)) - 1
            else:
                delta = 1 - (2 * (1 - u)) ** (1 / (eta + 1))

            lower = self.problem.variables_range[gene][0]
            upper = self.problem.variables_range[gene][1]

            if u < 0.5:
                child.features[gene] += delta * (child.features[gene] - lower)
            else:
                child.features[gene] += delta * (upper - child.features[gene])

            child.features[gene] = max(lower, min(upper, child.features[gene]))


class PortfolioEvolution:
    """
    Modified NSGA-II evolution loop for portfolio optimization.

    Integrates Lou 2023 adaptive mutation into the generation loop:
    after each generation, measures population diversity and adjusts
    the mutation distribution index accordingly.

    Args:
        problem: PortfolioProblem instance.
        num_of_generations: Number of evolutionary generations.
        num_of_individuals: Population size.
        num_of_tour_particips: Tournament size.
        tournament_prob: Tournament selection probability.
        crossover_param: SBX distribution index.
        mutation_param: Polynomial mutation distribution index.
        adaptive_mutation: Enable adaptive mutation (Lou 2023).
        smart_init: Enable smart initialization (Lou 2023).
        refined_selection: Enable refined selection (Lou 2023).
    """

    def __init__(self, problem, num_of_generations=200, num_of_individuals=100,
                 num_of_tour_particips=2, tournament_prob=0.9,
                 crossover_param=2, mutation_param=5,
                 adaptive_mutation=True, smart_init=True,
                 refined_selection=True):
        self.utils = PortfolioNSGA2Utils(
            problem, num_of_individuals, num_of_tour_particips,
            tournament_prob, crossover_param, mutation_param,
            adaptive_mutation=adaptive_mutation,
            smart_init=smart_init,
            refined_selection=refined_selection,
        )
        self.population = None
        self.num_of_generations = num_of_generations
        self.num_of_individuals = num_of_individuals

    def evolve(self):
        """
        Run the modified NSGA-II evolution loop.

        Returns:
            List of Individual objects on the final Pareto front (front 0).
        """
        # Initialize population (smart init if enabled)
        self.population = self.utils.create_initial_population()
        self.utils.fast_nondominated_sort(self.population)
        for front in self.population.fronts:
            self.utils.calculate_crowding_distance(front)
        children = self.utils.create_children(self.population)

        # Baseline diversity for adaptive mutation
        initial_diversity = self.utils.compute_population_diversity(self.population)

        returned_population = None
        for i in tqdm(range(self.num_of_generations), desc="NSGA-II"):
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

            # Partial front: take individuals with highest crowding distance
            self.utils.calculate_crowding_distance(self.population.fronts[front_num])
            self.population.fronts[front_num].sort(
                key=lambda ind: ind.crowding_distance, reverse=True
            )
            remaining = self.num_of_individuals - len(new_population)
            new_population.extend(self.population.fronts[front_num][:remaining])

            returned_population = self.population
            self.population = new_population

            # Lou 2023: Adapt mutation based on current diversity
            if self.utils.adaptive_mutation:
                diversity = self.utils.compute_population_diversity(self.population)
                self.utils.adapt_mutation_param(diversity, initial_diversity)

            # Prepare next generation
            self.utils.fast_nondominated_sort(self.population)
            for front in self.population.fronts:
                self.utils.calculate_crowding_distance(front)
            children = self.utils.create_children(self.population)

        return returned_population.fronts[0]
