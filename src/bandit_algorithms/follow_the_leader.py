import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI, Bandit_Algorithm_FI
import math

import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI, Bandit_Algorithm_FI
import math

class Follow_The_Leader_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.__label = "Follow The Leader"
        self.__data_generating_mechanism = data_generating_mechanism

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t):
        arm_estimates_current_round = np.mean(losses, axis = 1)
        return np.argmin(arm_estimates_current_round)
        
class Follow_The_Leader_PI(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.num_bootstrap_simulations = 100

        self.__label = "Follow The Leader"
        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K

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
        if (t < self.exploration_phase_length):
            A_t = math.floor(t / self.init_exploration)
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
            return A_t
        else:
            arm_estimates = np.sum(importance_weighted_losses, axis = 1)
            A_t = np.argmin(arm_estimates)
            p_t = np.zeros(shape = self.K) 
            p_t[A_t] = 1
            self.current_sampling_distribution = ((1 - (1 / t)) * p_t) + (1 / (t * self.K))
            
            # flip a bernoulli with success 
            # probability of (1/t) to add an 
            # explicit exploration component
            z = np.random.binomial(n = 1, p = (1 / t))

            if (z == 1):
                A_t = random.randint(low = 0, high = self.K, size = 1)
            
            return A_t