import numpy as np


class Individual(object):

    def __init__(self):
        self.rank = None
        self.crowding_distance = None
        self.domination_count = None
        self.dominated_solutions = None
        self.features = None
        self.objectives = None

    def __eq__(self, other):
        if isinstance(self, other.__class__):
            if isinstance(self.features, np.ndarray):
                return np.array_equal(self.features, other.features)
            return self.features == other.features

        return False

    def dominates(self, other_individual):
        """
        Vectorized dominance check using NumPy.
        (all objectives <=) AND (at least one objective <)
        """
        # Use NumPy broadcasting instead of Python loops
        self_obj = np.asarray(self.objectives)
        other_obj = np.asarray(other_individual.objectives)
        
        # All objectives must be <=
        all_leq = np.all(self_obj <= other_obj)
        # At least one objective must be strictly <
        any_less = np.any(self_obj < other_obj)
        
        return all_leq and any_less
