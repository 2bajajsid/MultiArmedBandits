import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

class Gaussian_FTPL_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism, num_arms, 
                 zero_mean = False, identity_cov = False, 
                 scaled_down = False, inflated_mean = False,
                 scaled_up = False):
        super().__init__()
        self.K = num_arms
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.zero_mean = zero_mean
        self.identity_cov = identity_cov
        self.scaled_down = scaled_down
        self.inflated_mean = inflated_mean
        self.scaled_up = scaled_up
        self.__label = "Gaussian FTPL {} {} {} {} {}".format("(zero-mean perturbation)" if self.zero_mean else "", 
                                                       "- (identity cov)" if self.identity_cov else "",
                                                       "- (scaled_down)" if self.scaled_down else "",
                                                       "- (inflated mean)" if self.inflated_mean else "", 
                                                       "- (scaled up)" if self.scaled_up else "")
        self.__data_generating_mechanism = data_generating_mechanism
        self.prob_distr = np.ones(self.K) / self.K


    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t, extra_param):
        if (t < self.exploration_phase_length):
            return math.floor(t / self.init_exploration)
        else:
            arm_estimates_current_round = np.mean(losses, axis = 1)

            if self.identity_cov == False:
                sample_cov_matrix = np.zeros(shape = (self.K, self.K))
                for i in range(t):
                    centered_loss_matrix = np.outer(losses[:, i] - arm_estimates_current_round, 
                                                    losses[:, i] - arm_estimates_current_round)
                    sample_cov_matrix += centered_loss_matrix
            else:
                sample_cov_matrix = t * np.eye(self.K)

            if self.scaled_down == True:
                sample_cov_matrix = (sample_cov_matrix / t)
            elif self.scaled_up == True:
                sample_cov_matrix = (self.T - t) * (sample_cov_matrix)

            rng = np.random.default_rng()
            num_counts_picked = np.zeros(shape = self.K)

            for i in range(200):

                if self.zero_mean == False:

                    if self.inflated_mean == True:
                        arm_estimates_current_round = t*arm_estimates_current_round

                    gaussian_perturbation = rng.multivariate_normal(arm_estimates_current_round, 
                                                                sample_cov_matrix, 
                                                                size = 1)
                else:
                    gaussian_perturbation = rng.multivariate_normal(np.zeros(shape = self.K), 
                                                                cov = sample_cov_matrix, 
                                                                size = 1)
                    
                num_counts_picked[np.argmin(arm_estimates_current_round + gaussian_perturbation)] += 1
            
            self.prob_distr = num_counts_picked / 200
            return np.random.choice(self.K, 
                                    p = self.prob_distr)
        
    def get_arm_distribution(self):
        return self.prob_distr