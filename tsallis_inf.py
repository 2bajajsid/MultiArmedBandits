import numpy as np
from numpy import random
from algorithm_class import Bandit_Algorithm_PI
import math

class Tsallis_Inf(Bandit_Algorithm_PI):

    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.x = 0
        self.label = "Tsallis_Inf"

    # Implementing Newton-Raphson helper for solving the
    # OMD optimization step as specified in 
    # Zimmert and Seldin (2022)
    def compute_p_t(self, cumulative_losses, neta_t):
     
        temp_x = np.inf 
        w_t = np.zeros(shape = len(cumulative_losses))

        while (abs(self.x - temp_x) > 0.0005):
            if (temp_x < np.inf):
                self.x = temp_x
            
            for i in range(len(cumulative_losses)):
                w_t[i] = 4 * (1 / (neta_t * (cumulative_losses[i] - self.x))**2)
            
            denom = (neta_t * sum(w_t ** (3/2)))
            temp_x = self.x - ((sum(w_t) - 1) / denom)

        self.x = temp_x
        return w_t

    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        neta_t = 2 * math.sqrt(1 / (t + 1))
        if (t < self.exploration_phase_length):
            return t / self.c
        else:
            cumulative_losses = np.sum(importance_weighted_losses, axis = 1)
            p_t = self.compute_p_t(cumulative_losses, neta_t)
            self.current_sampling_distribution = p_t / sum(p_t)
            a_t = np.random.choice(self.K, p = self.current_sampling_distribution, size = 1)
            return a_t

