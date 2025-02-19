import numpy as np
from numpy import random
from algorithm_class import Bandit_Algorithm_PI
import math

class Explore_Then_Commit(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.data_generating_mechanism = data_generating_mechanism
        self.current_sampling_distribution = np.ones(shape = data_generating_mechanism.get_K()) / data_generating_mechanism.get_K()
        self.label = "Explore-then-Commit"

    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        # first, explore then... 
        if (t <= self.data_generating_mechanism.get_exploration_phase_length()):
                A_t = math.floor(t / self.data_generating_mechanism.get_init_exploration())
        # ... commit
        else:
            arm_estimates_current_round = np.zeros(shape = self.data_generating_mechanism.get_K())
            for i in range(self.data_generating_mechanism.get_K()):
                arm_estimates_current_round[i] = np.mean(losses[i])

            A_t = np.argmin(arm_estimates_current_round)
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
        
        self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
        self.current_sampling_distribution[A_t] = 1
        return A_t