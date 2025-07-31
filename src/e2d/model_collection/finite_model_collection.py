from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract class for a finite model collection; 
# Each model class contains references to a finite number of models
class Finite_Model_Collection(ABC):

    def compute_instantaneous_regret(self, p_t, label, t):
        instantaneuous_regret = (self.models[self.M_star].arm_means[self.pi_star] - np.sum(np.multiply(p_t, self.models[self.M_star].arm_means)))
        # if (t % 100 == 0 and (label == "DEC (gamma): 2.0")):
        #    print(p_t)
        #    print("{0} at time {1}".format(instantaneuous_regret, t))
        #    print("*****")
        return instantaneuous_regret
    
    # draws a [K x m] Monte-Carlo sample 
    def draw_sample_from_model_index(self, model_index, sample_size):
        sample_drawn = np.zeros(shape = (self.K, sample_size))
        for m in range(sample_size):
            sample_drawn[:, m] = self.models[model_index].generate_observation()
        return sample_drawn
    
    def get_delta_min(self):
        delta_min = np.inf
        for i in range(len(self.models)):
            delta_min = np.min([delta_min, self.models[i].get_delta()])
        return delta_min

    def get_delta_delta_min(self):
        delta_min_min = np.inf
        for i in range(1, len(self.models)):
            arm_means_i_minus_1 = self.models[i-1].arm_means 
            arm_means_i = self.models[i].arm_means
            arm_means_diff = np.min(np.abs(arm_means_i_minus_1 - arm_means_i))
            delta_min_min = np.min([arm_means_diff, delta_min_min])
        return delta_min_min
    
    # [1 x K] squared hellinger divergence matrix
    def compute_true_sq_hellinger_divergence(self, model_index_i, model_index_j):
        model_i_mean = self.models[model_index_i].arm_means
        model_i_sd = 0.5 * np.ones(shape = self.K)

        model_j_mean = self.models[model_index_j].arm_means
        model_j_sd = 0.5 * np.ones(shape = self.K)

        true_hellinger_dist = np.zeros(shape = self.K)
        for a in range(self.K):
            mu_1_minus_mu_2 = (model_i_mean[a] - model_j_mean[a])**2
            denom = (model_i_sd[a]**2) + (model_j_sd[a]**2)
            true_hellinger_dist[a] = 1 - (np.sqrt((2 * model_i_sd[a] * model_j_sd[a]) / denom) * np.exp(-0.25 * mu_1_minus_mu_2/denom))

        return true_hellinger_dist
    
    def print_description(self):
        for i in range(self.M):
            description = "Model {0} ".format(i)
            for j in range(self.K):
                description += "Arm {1} has Mean {2}".format(i, j, self.models[i].arm_means[j]) + "  "
            print(description)

    def get_model_class_length(self):
        return self.M
    
    def get_rt(self, action_pi):
        return self.models[self.M_star].generate_observation()[action_pi]