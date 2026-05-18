import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Low_Rank_Stochastic(Data_Generating_Mechanism):
        
    def __init__(self, num_arms = 20, 
                 init_exploration = 1, 
                 optimality_gap = 0.1, 
                 low_rank_dimension = 10, 
                 M = 100):
        
        mu_arms = np.random.uniform(low = 0.25, 
                                    high = 1.5, 
                                    size = low_rank_dimension)
        
        if (low_rank_dimension > 1):
            self.second_optimal_arm_index = np.flip(np.argsort(mu_arms))[1]
            self.optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
            mu_arms[self.optimal_arm_index] = mu_arms[self.second_optimal_arm_index] + 0.3333

        self.low_rank_subspace = np.random.rand(num_arms, 
                                                low_rank_dimension)
        self.actual_num_arms = num_arms
        time = int((np.log(num_arms) / (0.02**2))) * 2

        super().__init__(time_horizon = time, 
                         mu_arms = mu_arms, 
                         num_runs = M, 
                         init_exploration = init_exploration)
        
    def get_K(self):
        return self.actual_num_arms
        
    def get_rewards(self, t):
        rng = np.random.default_rng()
        rewards = np.matmul(self.low_rank_subspace, 
                            rng.multivariate_normal(mean = self.get_mu_arms(), 
                                                    cov = np.eye(np.shape(self.get_mu_arms())[0]),
                                                    size = 1)[0])
        return rewards