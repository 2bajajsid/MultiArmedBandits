import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism
from scipy.linalg import hadamard

class Hadamard_Adversarial(Data_Generating_Mechanism):
        
    def __init__(self, num_arms = 8, M = 100):
        mu_arms = np.zeros(shape = num_arms)
        super().__init__(time_horizon = 900, 
                         mu_arms = mu_arms, 
                         num_runs = M,
                         init_exploration=5)
        self.h_matrix = (hadamard(1024, dtype = int) + 1) / 2 

    def get_rewards(self, t):
        return self.h_matrix[:self.get_K(),
                             t]
            