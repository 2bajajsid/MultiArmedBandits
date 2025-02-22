import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Low_Gap_Stochastic(Data_Generating_Mechanism):
        
    def __init__(self, num_arms = 50):
        mu_arms = np.random.uniform(low = 0.25, high = 0.75, size = num_arms)
        optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
        second_optimal_arm_index = np.flip(np.argsort(mu_arms))[1]
        mu_arms[second_optimal_arm_index] = mu_arms[optimal_arm_index] - 0.05
        super().__init__(time_horizon = 3000, mu_arms = mu_arms, num_runs = 100, init_exploration = 5)

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.get_K())

        for i in range(self.get_K()):
            rewards[i] = random.binomial(n = 1, p = self.get_mu_arm_i(i))

        return rewards