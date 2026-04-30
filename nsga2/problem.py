import numpy as np

from nsga2.individual import Individual


class Problem:
    def __init__(
        self,
        objectives,
        num_of_variables,
        variables_range,
        expand=True,
        same_range=False,
    ):
        self.num_of_objectives = len(objectives)
        self.num_of_variables = num_of_variables
        self.objectives = objectives
        self.expand = expand
        if same_range:
            self.variables_range = [variables_range[0] for _ in range(num_of_variables)]
        else:
            self.variables_range = variables_range

    def generate_individual(self):
        individual = Individual()
        individual.features = np.array(
            [np.random.uniform(low, high) for low, high in self.variables_range],
            dtype=np.float64,
        )
        return individual

    def calculate_objectives(self, individual):
        if self.expand:
            individual.objectives = np.array(
                [f(*individual.features) for f in self.objectives],
                dtype=np.float64,
            )
            return

        individual.objectives = np.array(
            [f(individual.features) for f in self.objectives],
            dtype=np.float64,
        )
