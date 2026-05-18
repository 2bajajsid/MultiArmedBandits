import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Stochastically_Constrained_Fast_Switch(Data_Generating_Mechanism):

    def __init__(self, num_arms = 25, time_horizon = 2000, init_exploration = 1):
        mu_arms = np.random.uniform(low = 0.25, high = 2.0, size = num_arms)
        self.optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
        self.worst_arm_index = np.argsort(mu_arms)[0]
        super().__init__(time_horizon = time_horizon, 
                    mu_arms = mu_arms, 
                    num_runs = 100, 
                    init_exploration = init_exploration)
        A = np.random.rand(self.get_K(), self.get_K())
        self.vcov = np.dot(A, A.T) + np.identity(self.get_K())

    def get_rewards(self, t):
        rng = np.random.default_rng()
        rewards = rng.multivariate_normal(self.get_mu_arms(),  
                                          cov = self.vcov, 
                                          size = 1)[0]

        # switch up rewards every 3rd time-step
        if (t%3 == 1):
            temp = rewards[self.optimal_arm_index]
            rewards[self.optimal_arm_index] = rewards[self.worst_arm_index]
            rewards[self.worst_arm_index] = temp

        return rewards