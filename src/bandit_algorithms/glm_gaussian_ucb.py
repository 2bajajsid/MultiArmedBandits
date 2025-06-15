import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI
import math

class GLM_Gaussian_UCB(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism, label):
        super().__init__()
        self.__data_generating_mechanism = data_generating_mechanism
        self.__label = label
        self.__current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())

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
    
    def get_arm_to_pull(self, t):
        if t < 60:
            return int(np.random.uniform(0, 100, size = 1))
        else:
            arm_estimates_current_round = np.zeros(shape = self.data_generating_mechanism.get_K())
            for j in range(self.data_generating_mechanism.get_K()):
                arm_estimates_current_round[j] = self.data_generating_mechanism.get_arm_index(j, t)
            A_t = np.argmax(arm_estimates_current_round)

            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
            return A_t