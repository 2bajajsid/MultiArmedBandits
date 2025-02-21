import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI
import math

class UCB(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.__data_generating_mechanism = data_generating_mechanism
        self.__label = "UCB"
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

    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        # first, explore then ... 
        if (t < self.data_generating_mechanism.get_exploration_phase_length()):
            A_t = math.floor(t / self.data_generating_mechanism.get_init_exploration())
        else:
            arm_estimates_current_round = np.zeros(shape = self.data_generating_mechanism.get_K())
            for j in range(self.data_generating_mechanism.get_K()):
                s = len(losses[j])
                f_hat_pi = (1 - np.mean(losses[j]))
                f_t = 1 + (t * (math.log(t)**2))
                half_confidence_interval_width = math.sqrt((2 * math.log(f_t)) / (s))
                arm_estimates_current_round[j] = f_hat_pi + half_confidence_interval_width
            A_t = np.argmax(arm_estimates_current_round)
        
        self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
        self.current_sampling_distribution[A_t] = 1
        return A_t