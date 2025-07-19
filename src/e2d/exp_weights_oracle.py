from abc import ABC, abstractmethod
import numpy as np
from numpy import random
import math

# Estimation Oracle class
# is responsible for using some sort of 
# a weighting algorithm and returns the index of 
# the expert that has been most accurate so far
class Exp_Weights_Oracle(ABC):
    def __init__(self, T, M):
        super().__init__()
        self.M = M
        self.T = T
        self.accumulated_losses = np.zeros(shape = self.M)
        self.neta = math.sqrt((2 * math.log(self.M)) / self.T)

    @abstractmethod
    def get_m_hat_index(self):
        prob_distr = np.zeros(shape = self.M) 
        for i in range(self.M):
            prob_distr[i] = math.exp(-1 * self.neta * self.accumulated_losses[i])
        normalization_constant = np.sum(prob_distr)
        prob_distr = self.cumulative_losses / normalization_constant 
        return np.random.choice(self.M, p = prob_distr)

    @abstractmethod
    def add_to_training_data_set(self, o_t, r_t):
        self.training_observations.append(o_t)
        self.training_y.append(r_t)
        for i in range(self.M):
            self.accumulated_losses[i] += (r_t - o_t[i])

    @abstractmethod
    def clear(self):
        self.accumulated_losses = np.zeros(shape = self.M)