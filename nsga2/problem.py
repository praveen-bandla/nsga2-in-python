import numpy as np

from nsga2.individual import Individual


class Problem:

    def __init__(self, objectives, num_of_variables, variables_range, expand=True, same_range=False):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.expand = expand
        self.variables_range = []
        if same_range:
            for _ in range(num_of_variables):
                self.variables_range.append(variables_range[0])
        else:
            self.variables_range = variables_range

    def generate_individual(self):
        """Generate individual with features as numpy array (not list)."""
        individual = Individual()
        # Use NumPy random (vectorized) instead of list comprehension
        features = np.array(
            [np.random.uniform(low, high) for low, high in self.variables_range],
            dtype=np.float64
        )
        individual.features = features  # Store as array directly
        return individual

    def calculate_objectives(self, individual):
        """Evaluate objectives. PortfolioProblem overrides this."""
        if self.expand:
            individual.objectives = np.array(
                [f(*individual.features) for f in self.objectives],
                dtype=np.float64
            )
        else:
            individual.objectives = np.array(
                [f(individual.features) for f in self.objectives],
                dtype=np.float64
            )
