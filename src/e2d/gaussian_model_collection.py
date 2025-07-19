import numpy as np
from numpy import random
from e2d.finite_model_collection import Finite_Model_Collection
from e2d.gaussian_model_class import Gaussian_Model_Class
import math

class Gaussian_Model_Collection(Finite_Model_Collection):
    def __init__(self, M = 10, K = 2):
        super().__init__()
        self.M = M
        self.K = K
        self.models = []
        for i in range(M):
            self.models.append(Gaussian_Model_Class(self.K))
        self.M_star = np.randint(self.M)
        self.pi_star = self.models[self.M_star].get_optimal_arm_index()
    
    def get_model_class_length(self):
        return self.M
    
    def get_ot(self, action_pi):
        self.o_t = np.zeros(shape = (self.M, self.K))
        for i in range(self.M):
            self.ot[i:, ] = self.models[i].generate_observation()
        return self.o_t[:, action_pi]
    
    def compute_instantaneous_regret(self, p_t):
        return (self.models[self.M_star].arm_means[self.pi_star] - (p_t * self.ot[self.M_star, :]))