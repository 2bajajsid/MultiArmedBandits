import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class High_Gap_Stochastic(Data_Generating_Mechanism):
        
    def __init__(self, num_arms = 25, time_horizon = 2000, init_exploration = 1):
        mu_arms = np.random.uniform(low = 0.25, high = 0.6, size = num_arms)
        optimal_arm_index = np.argmax(mu_arms)
        mu_arms[optimal_arm_index] = 0.8
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = mu_arms, 
                         num_runs = 100, 
                         init_exploration = init_exploration)

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.get_K())

        for i in range(self.get_K()):
            rewards[i] = random.binomial(n = 1, p = self.get_mu_arm_i(i))

        return rewards