from abc import ABC, abstractmethod
import numpy as np
from numpy import random
from e2d.estimation_oracle import Estimation_Oracle
from scipy.special import softmax
import math

# Estimation Oracle class
# is responsible for using some sort of 
# a weighting algorithm and returns the index of 
# the expert that has been most accurate so far
class Exp_Weights_Oracle(Estimation_Oracle):
    def __init__(self, T, M):
        super().__init__()
        self.M = M
        self.T = T
        self.accumulated_losses = np.zeros(shape = self.M)
        self.neta = math.sqrt((8 * math.log(self.M)) / self.T)
        self.p = (np.ones(shape = self.M) / self.M)

    def get_m_hat_index(self):
        self.p = softmax(- self.neta * self.accumulated_losses)
        return np.random.choice(self.M, p = self.p)

    def add_to_training_data_set(self, f_m_hat, r_t, p_t):
        for i in range(self.M):
            p_t = 1
            self.accumulated_losses[i] += ((f_m_hat[i] - r_t)**2 / p_t)
        #print(self.accumulated_losses)

    def clear(self):
        self.accumulated_losses = np.zeros(shape = self.M)
        self.p = (np.ones(shape = self.M) / self.M)

    def current_sampling_distribution(self):
        return self.p