import numpy as np
import math
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class High_Gap_Stochastic(Data_Generating_Mechanism):
        
    def __init__(self, num_arms = 25, init_exploration = 1, optimality_gap = 0.333, M = 100):
        mu_arms = np.random.uniform(low = 0.1, high = 0.5, size = num_arms)
        optimal_arm_index = np.argmax(mu_arms)
        mu_arms[optimal_arm_index] = 0.95
        
        self.second_optimal_arm_index = np.flip(np.argsort(mu_arms))[1]
        self.optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
        mu_arms[self.optimal_arm_index] = mu_arms[self.second_optimal_arm_index] + optimality_gap

        time_horizon = 1000

        A = np.random.rand(num_arms, num_arms)
        self.vcov = np.dot(A, A.T)
        self.vcov += (np.eye(num_arms) * 1e-6)
        
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = mu_arms, 
                         num_runs = M, 
                         init_exploration = init_exploration)

    def get_rewards(self, t):
        rng = np.random.default_rng()
        rewards = rng.multivariate_normal(self.get_mu_arms(),  
                                          cov = self.vcov, 
                                          size = 1)[0]
        return rewards