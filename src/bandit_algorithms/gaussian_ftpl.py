import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

MEAN_SCALE_ZERO_MEAN = 0
MEAN_SCALE_SAMPLE_MEAN = 1
MEAN_SCALE_BAGGING_FROM_PAST = 2

COVARIANCE_SCALE_SAMPLE = 0
COVARIANCE_SCALE_OUTER_GRAM = 1
COVARIANCE_SCALE_BAGGING_FROM_PAST = 2

IDENTITY_COVARIANCE = 0
SAMPLE_COVARIANCE = 1

class Gaussian_FTPL_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism, num_arms, mean_scale, covariance_scale, covariance_type):
        super().__init__()
        self.K = num_arms
        self.T = data_generating_mechanism.get_T()
        
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        
        self.mean_scale = mean_scale
        self.covariance_scale = covariance_scale
        self.covariance_type = covariance_type

        mean_scale_label = ""
        if mean_scale == MEAN_SCALE_ZERO_MEAN:
            mean_scale_label = "zero-mean"
        elif mean_scale == MEAN_SCALE_SAMPLE_MEAN:
            mean_scale_label = "sample-mean"
        else:
            mean_scale_label = "(T-t) * sample-mean"

        covariance_scale_label = ""
        if covariance_scale == COVARIANCE_SCALE_SAMPLE and covariance_type == IDENTITY_COVARIANCE:
            covariance_scale_label = "identity-covariance"
        elif covariance_scale == COVARIANCE_SCALE_SAMPLE and covariance_type == SAMPLE_COVARIANCE:
            covariance_scale_label = "sample-covariance"
        elif covariance_scale == COVARIANCE_SCALE_OUTER_GRAM and covariance_type == IDENTITY_COVARIANCE:
            covariance_scale_label = "t * identity"
        elif covariance_scale == COVARIANCE_SCALE_OUTER_GRAM and covariance_type == SAMPLE_COVARIANCE:
            covariance_scale_label = "t * sample-covariance"
        elif covariance_scale == COVARIANCE_SCALE_BAGGING_FROM_PAST and covariance_type == IDENTITY_COVARIANCE:
            covariance_scale_label = "((T-t)^2 / t) * identity"
        else:
            covariance_scale_label = "((T-t)^2 / t) * sample-covariance"

        self.__label = "Gaussian-FTPL ({}) ({})".format(mean_scale_label, 
                                                    covariance_scale_label)

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
            losses_sum_current_round = np.sum(losses, axis = 1)
        
            if self.mean_scale == MEAN_SCALE_ZERO_MEAN:
                perturbation_mean = np.zeros(shape = self.K)
            elif self.mean_scale == MEAN_SCALE_SAMPLE_MEAN:
                perturbation_mean = np.mean(losses, axis = 1)
            else:
                perturbation_mean = (self.T-t) * np.mean(losses, axis = 1)

            if self.covariance_type == IDENTITY_COVARIANCE:
                if self.covariance_scale == COVARIANCE_SCALE_SAMPLE:
                    covariance_matrix = np.eye(self.K)
                elif self.covariance_scale == COVARIANCE_SCALE_OUTER_GRAM:
                    covariance_matrix = t * np.eye(self.K)
                else:
                    covariance_matrix = ((self.T - t)**2 / t) * np.eye(self.K)
            else:
                outer_gram_product = np.zeros(shape = (self.K, self.K))
                rewards_estimates_current_round = (1 - np.mean(losses, axis = 1))
                for i in range(t):
                    centered_reward_outer_product = np.outer((1 - losses[:, i]) - rewards_estimates_current_round, 
                                                    (1 - losses[:, i]) - rewards_estimates_current_round)
                    outer_gram_product += centered_reward_outer_product
                
                if self.covariance_scale == COVARIANCE_SCALE_SAMPLE:
                    covariance_matrix = (1 / t) * outer_gram_product
                elif self.covariance_scale == COVARIANCE_SCALE_OUTER_GRAM:
                    covariance_matrix = outer_gram_product
                else:
                    covariance_matrix = ((self.T - t)**2 / t) * outer_gram_product

            rng = np.random.default_rng()
            num_counts_picked = np.zeros(shape = self.K)

            for i in range(200):
                gaussian_perturbation = rng.multivariate_normal(perturbation_mean, 
                                                                covariance_matrix, 
                                                                size = 1)
                num_counts_picked[np.argmin(losses_sum_current_round + gaussian_perturbation)] += 1
            
            self.prob_distr = num_counts_picked / 200
            return np.random.choice(self.K, 
                                    p = self.prob_distr)
        
    def get_arm_distribution(self):
        return self.prob_distr