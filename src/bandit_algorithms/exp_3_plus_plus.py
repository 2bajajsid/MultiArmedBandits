import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI
import math

class Exp3_plus_plus(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "Exp3++"

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label
    
    @property
    def current_sampling_distribution(self):
        return self.__current_sampling_distribution
    
    @current_sampling_distribution.setter
    def current_sampling_distribution(self, distr):
        self.__current_sampling_distribution = distr

    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        if (t <= self.exploration_phase_length):
            A_t = math.floor(t / self.init_exploration)
        else:
            alpha = 3
            beta = 256
            # data structures that mimics
            # the pseudo-code of Algorithm 2: Gap  
            # Estimation in Randomized Playing Strategies
            eta_t = np.zeros(shape = self.K)
            epsilon_t = np.zeros(shape = self.K)
            N_t_minus_one = np.zeros(shape = self.K)
            UCB_t = np.zeros(shape = self.K)
            LCB_t = np.zeros(shape = self.K)
            delta_t = np.zeros(shape = self.K)

            for a in range(self.K):
                N_t_minus_one[a] = len(losses[a])

            neta_t = 1/2 * (math.sqrt(math.log(self.K) / t * self.K))
            cumulative_losses = np.sum(importance_weighted_losses, axis = 1)
            for a in range(self.K):
                cumulative_losses[a] = math.exp(-1 * neta_t * cumulative_losses[a])
                margin_confidence_interval = math.sqrt((alpha * math.log(t * (self.K)**(1 / alpha))) / 2 * N_t_minus_one[a])
                UCB_t[a] = min(1, (cumulative_losses[a] / N_t_minus_one[a]) + margin_confidence_interval)
                LCB_t[a] = max(0, (cumulative_losses[a] / N_t_minus_one[a]) - margin_confidence_interval)

            for a in range(self.K):
                delta_t[a] = max(0.00001, LCB_t[a] - min(UCB_t))
                eta_t[a] = (beta * math.log(t)) / (t * (delta_t[a])**2)
                epsilon_t[a] = min(1 / (2*self.K), 
                                   (1 / 2) * (math.sqrt(math.log(self.K) / 
                                                        (t * self.K))), eta_t[a])

            normalization_constant = np.sum(cumulative_losses)
            self.current_sampling_distribution = cumulative_losses / normalization_constant

            sum_epsilon_t = np.sum(epsilon_t)

            for a in range(self.K):
                self.current_sampling_distribution[a] = ((1 - sum_epsilon_t) * self.current_sampling_distribution[a]) + (epsilon_t[a])

            A_t = np.random.choice(self.K, p = self.current_sampling_distribution)
    
        return A_t