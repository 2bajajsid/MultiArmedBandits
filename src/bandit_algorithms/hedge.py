import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

class Hedge(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.neta = math.sqrt((2 * math.log(self.K)) / self.T)
        self.__label = "Hedge"
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
            cumulative_losses = np.sum(losses, axis = 1)
            for i in range(self.K):
                cumulative_losses[i] = math.exp(-1 * self.neta * cumulative_losses[i])

            normalization_constant = np.sum(cumulative_losses)
            prob_distr = cumulative_losses / normalization_constant 
            self.prob_distr = prob_distr
            return np.random.choice(self.K, p = self.prob_distr)
        
    def get_arm_distribution(self):
        return self.prob_distr
