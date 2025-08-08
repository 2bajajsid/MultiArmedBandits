import numpy as np
from numpy import random
from e2d.model_collection.finite_model_collection import Finite_Model_Collection
from e2d.model_class.poisson_model_class import Poisson_Model_Class
from e2d.model_class.gaussian_model_class import Gaussian_Model_Class
from e2d.model_class.student_t_model_class import Student_t_Model_Class
from e2d.model_class.cauchy_model_class import Cauchy_Model_Class
import math

class Mixed_Model_Collection(Finite_Model_Collection):
    def __init__(self, M = 10, K = 2, Optimality_Gap = 0.25):
        super().__init__()
        self.M = M
        self.K = K
        self.models = []
        for i in range(M):
            if (i % 4 == 0):
                self.models.append(Student_t_Model_Class(K = self.K, Delta = Optimality_Gap))
            elif (i % 4 == 1):
                self.models.append(Gaussian_Model_Class(K = self.K, Delta = Optimality_Gap))
            elif (i % 4 == 2):
                self.models.append(Student_t_Model_Class(K = self.K, Delta = Optimality_Gap))
            else:
                self.models.append(Gaussian_Model_Class(K = self.K, Delta = Optimality_Gap))
        self.M_star = np.random.randint(self.M)
        self.pi_star = self.models[self.M_star].get_optimal_arm_index()
        self.t = 0