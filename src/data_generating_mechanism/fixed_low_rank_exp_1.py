import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Fixed_Low_Rank_Exp_1(Data_Generating_Mechanism):
        
    def __init__(self, M = 100, gap = 0.45):
        
        mu_arms = [0.5, 0.5, 0.5, 0.5, 
                   0.5 + gap, 0.5 + gap, 0.5 + gap, 0.5 + gap]
        self.sigma_cov = np.array([
                        [1, 0.6, 0.6, 0.6, -0.2, -0.2, -0.2, -0.2],
                        [0.6, 1, 0.6, 0.6, -0.2, -0.2, -0.2, -0.2],
                        [0.6, 0.6, 1, 0.6, -0.2, -0.2, -0.2, -0.2],
                        [0.6, 0.6, 0.6, 1, -0.2, -0.2, -0.2, -0.2],
                        [-0.2, -0.2, -0.2, -0.2, 1, 0.6, 0.6, 0.6],
                        [-0.2, -0.2, -0.2, -0.2, 0.6, 1, 0.6, 0.6],
                        [-0.2, -0.2, -0.2, -0.2, 0.6, 0.6, 1, 0.6],
                        [-0.2, -0.2, -0.2, -0.2, 0.6, 0.6, 0.6, 1]
                    ])
        
        time = 5000

        super().__init__(time_horizon = time, 
                         mu_arms = mu_arms, 
                         num_runs = M, 
                         init_exploration = 1)
        
    def get_K(self):
        return 8
        
    def get_rewards(self, t):
        rng = np.random.default_rng()
        rewards = rng.multivariate_normal(mean = self.get_mu_arms(), 
                                          cov = self.sigma_cov,
                                          size = 1)[0]
        return rewards