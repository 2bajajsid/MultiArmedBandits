import numpy as np
from numpy import random
from algorithm_class import Bandit_Algorithm_FI
import math

class Hedge(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.neta = math.sqrt((2 * math.log(self.K)) / self.T)
        self.__label = "Hedge"
        self.__data_generating_mechanism = data_generating_mechanism

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t):
        cumulative_losses = np.sum(losses, axis = 1)
        for i in range(self.K):
            cumulative_losses[i] = math.exp(-1 * self.neta * cumulative_losses[i])

        normalization_constant = np.sum(cumulative_losses)
        prob_distr = cumulative_losses / normalization_constant 

        return np.random.choice(self.K, p = prob_distr)