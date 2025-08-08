import numpy as np
from numpy import random
from e2d.model_collection.finite_model_collection import Finite_Model_Collection
from e2d.model_class.gaussian_model_class import Gaussian_Model_Class
import math

class Gaussian_Model_Collection(Finite_Model_Collection):
    def __init__(self, M = 3, K = 5, Optimality_Gap = 0.25):
        super().__init__()
        self.M = M
        self.K = K
        self.models = []
        self.models.append(Gaussian_Model_Class(K = self.K, Delta = Optimality_Gap))
        for i in range(1, self.M):
            self.models.append(Gaussian_Model_Class(K = self.K, Delta = Optimality_Gap, arm_means = self.models[i-1].arm_means))
        
        self.mean_matrix = np.zeros(shape = (M, K))
        self.mean_matrix[0, :] = [0.8, 0.2, 0.4, 0.6, 0.0]
        self.mean_matrix[1, :] = [0.3, 0.9, 0.55, 0.8, 0.1]
        self.mean_matrix[2, :] = [0.45, 0.65, 1.0, 0.75, 0.3]
        self.multipliers = [-1, 1]
        self.multiplier = self.multipliers[np.random.randint(len(self.multipliers), size=1)[0]]

        for m in range(self.M):
            for k in range(self.K):
                self.models[m].arms[k].f_m = self.multiplier * self.mean_matrix[m][k]
                self.models[m].arm_means[k] = self.multiplier * self.mean_matrix[m][k]

        self.M_star = np.random.randint(self.M)
        self.pi_star = self.models[self.M_star].get_optimal_arm_index()
        self.t = 0

    def compute_true_sq_hellinger_divergence(self, model_index_i, model_index_j):
        model_i_mean = self.models[model_index_i].arm_means
        model_i_sd = 0.5 * np.ones(shape = self.K)

        model_j_mean = self.models[model_index_j].arm_means
        model_j_sd = 0.5 * np.ones(shape = self.K)

        true_hellinger_dist = np.zeros(shape = self.K)
        for a in range(self.K):
            mu_1_minus_mu_2 = (model_i_mean[a] - model_j_mean[a])**2
            denom = (model_i_sd[a]**2) + (model_j_sd[a]**2)
            true_hellinger_dist[a] = 1 - (np.sqrt(((2 * model_i_sd[a] * model_j_sd[a]) / denom)) * np.exp(-0.25 * mu_1_minus_mu_2/denom))

        return true_hellinger_dist
    
    def get_squared_hellinger_distance_lower_bound(self, optimality_gap, sigma_p=0.5, sigma_q=0.5):
        return (1 - np.sqrt(1 - (optimality_gap**2 / (optimality_gap**2 + (sigma_p + sigma_q)**2))))
    
    def compute_true_radon_nikodym_derivative(self, model_index_i, model_index_j, x, lambda_mixture = 1/2):
        model_i_mean = self.models[model_index_i].arm_means
        model_j_mean = self.models[model_index_j].arm_means

        dp_over_dr = np.zeros(shape = self.K)
        dq_over_dr = np.zeros(shape = self.K)
        for a in range(self.K):
            mu_q = model_i_mean[a]
            mu_p = model_j_mean[a]
            mu_r = (lambda_mixture * mu_p) + ((1 - lambda_mixture) * mu_q)
            dp_over_dr[a] = np.exp(-1 * ((x - mu_p)**2 - (x - mu_r)**2) / 2)
            dq_over_dr[a] = np.exp(-1 * ((x - mu_q)**2 - (x - mu_r)**2) / 2)
        
        radon_nikodym = np.zeros(shape = (2, self.K))
        radon_nikodym[0, :] = dp_over_dr
        radon_nikodym[1, :] = dq_over_dr

        return radon_nikodym

    def compute_true_mean_square_divergence(self, model_index_i, model_index_j):
        true_mean_square_dist = np.zeros(shape = self.K)
        for a in range(self.K):
            #print(self.models[model_index_i].arm_means[a] - self.models[model_index_j].arm_means[a])
            true_mean_square_dist[a] = (self.models[model_index_i].arm_means[a] - self.models[model_index_j].arm_means[a])**2
        return true_mean_square_dist
    