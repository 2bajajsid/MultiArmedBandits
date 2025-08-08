import numpy as np
from numpy import random
from e2d.model_collection.finite_model_collection import Finite_Model_Collection
from e2d.model_class.bernoulli_model_class import Bernoulli_Model_Class
import math

class Bernoulli_Model_Collection(Finite_Model_Collection):
    def __init__(self, M = 10, K = 2, Optimality_Gap = 0.25):
        super().__init__()
        self.M = M
        self.K = K
        self.models = []
        self.models.append(Bernoulli_Model_Class(K = self.K, Delta = Optimality_Gap))
        for i in range(1, self.M):
            self.models.append(Bernoulli_Model_Class(K = self.K, Delta = Optimality_Gap, arm_means = self.models[i-1].arm_means))

        self.mean_matrix = np.zeros(shape = (M, K))
        self.mean_matrix[0, :] = [0.8, 0.2, 0.4, 0.6, 0.0]
        self.mean_matrix[1, :] = [0.3, 0.9, 0.55, 0.8, 0.1]
        self.mean_matrix[2, :] = [0.45, 0.65, 1.0, 0.75, 0.3]
        self.multipliers = [1.0]
        self.multiplier = self.multipliers[np.random.randint(len(self.multipliers), size=1)[0]]

        for m in range(self.M):
            for k in range(self.K):
                self.models[m].arms[k].f_m = 1.0 * self.mean_matrix[m][k]
                self.models[m].arm_means[k] = 1.0 * self.mean_matrix[m][k]

        self.M_star = np.random.randint(self.M)
        self.pi_star = self.models[self.M_star].get_optimal_arm_index()
        self.t = 0

    def compute_true_sq_hellinger_divergence(self, model_index_i, model_index_j):
        model_i_mean = self.models[model_index_i].arm_means
        model_j_mean = self.models[model_index_j].arm_means
        true_hellinger_dist = np.zeros(shape = self.K)
        for a in range(self.K):
            p_0 = (1 - model_i_mean[a])
            p_1 = model_i_mean[a]
            
            q_0 = (1 - model_j_mean[a])
            q_1 = model_j_mean[a]

            true_hellinger_dist[a] = (1 / np.sqrt(2)) * ((np.sqrt(p_0) - np.sqrt(q_0))**2 +
                                                          (np.sqrt(p_1) - np.sqrt(q_1))**2)

        return true_hellinger_dist
    
    def compute_true_mean_square_divergence(self, model_index_i, model_index_j):
        true_mean_square_dist = np.zeros(shape = self.K)
        for a in range(self.K):
            true_mean_square_dist[a] = (self.models[model_index_i].arm_means[a] - self.models[model_index_j].arm_means[a])**2
        return true_mean_square_dist