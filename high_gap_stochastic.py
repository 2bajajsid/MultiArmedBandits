import numpy as np
from numpy import random
from data_generating_mechanism import Data_Generating_Mechanism

class High_Gap_Stochastic(Data_Generating_Mechanism):
        
    def __init__(self):
        mu_arms = np.random(low = 0.25, high = 0.5, size = 10)
        optimal_arm_index = np.argmax(mu_arms)
        mu_arms[optimal_arm_index] = 0.8
        super().__init__(time_horizon = 1000, mu_arms = mu_arms, num_runs = 1000, init_exploration = 5)

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.__num_arms)

        for i in range(self.__num_arms):
            rewards[i] = random.binomial(n = 1, p = self.__mu_arms[i])

        return rewards