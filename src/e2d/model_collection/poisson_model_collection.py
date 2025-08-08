import numpy as np
from numpy import random
from e2d.model_collection.finite_model_collection import Finite_Model_Collection
from e2d.model_class.poisson_model_class import Poisson_Model_Class
import math

class Poisson_Model_Collection(Finite_Model_Collection):
    def __init__(self, M = 10, K = 2, Optimality_Gap = 0.25):
        super().__init__()
        self.M = M
        self.K = K
        self.models = []
        for i in range(M):
            self.models.append(Poisson_Model_Class(K = self.K, Delta = Optimality_Gap))
        self.M_star = np.random.randint(self.M)
        self.pi_star = self.models[self.M_star].get_optimal_arm_index()
        self.t = 0