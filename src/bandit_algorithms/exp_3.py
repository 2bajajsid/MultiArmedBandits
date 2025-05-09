import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI
import math

class Exp3(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = data_generating_mechanism.get_K()) / data_generating_mechanism.get_K()
        self.__label = "Exp3"

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
        neta = math.sqrt((2 * math.log(self.data_generating_mechanism.get_K())) / 
                         (self.data_generating_mechanism.get_K() * self.data_generating_mechanism.get_T()))
        cumulative_losses = np.sum(importance_weighted_losses, axis = 1)
        
        for i in range(self.data_generating_mechanism.get_K()):
            cumulative_losses[i] = math.exp(-1 * neta * cumulative_losses[i])

        normalization_constant = np.sum(cumulative_losses)
        self.current_sampling_distribution = cumulative_losses / normalization_constant
        
        A_t = np.random.choice(self.data_generating_mechanism.get_K(), p = self.current_sampling_distribution)
        return A_t