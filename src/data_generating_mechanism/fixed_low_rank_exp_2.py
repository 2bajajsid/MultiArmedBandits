import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Fixed_Low_Rank_Exp_2(Data_Generating_Mechanism):
        
    def __init__(self, M = 100, gap = 0.1):
        
        mu_arms = [0.5, 0.5 + gap]
        self.A = np.array([
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [1, 0],
                        [0, 1],
                        [0, 1],
                        [0, 1],
                        [0, 1]
                    ])
        self.vcov = np.array([
            [1, -0.5],
            [-0.5, 1]
        ])
        time = 2500

        super().__init__(time_horizon = time, 
                         mu_arms = mu_arms, 
                         num_runs = M, 
                         init_exploration = 1)
        
    def get_K(self):
        return 8
        
    def get_rewards(self, t):
        rng = np.random.default_rng()
        latent_factors = rng.multivariate_normal(mean = self.get_mu_arms(), 
                                                 cov = self.vcov, 
                                                 size = 1)[0]
        rewards = np.matmul(self.A, latent_factors) + rng.multivariate_normal(mean = np.zeros(shape = 8),
                                                                              cov = np.eye(8),
                                                                              size = 1)[0]
        return rewards