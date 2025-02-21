import numpy as np
from numpy import random
from data_generating_mechanism import Data_Generating_Mechanism

class Stochastically_Constrained(Data_Generating_Mechanism):

    def __init__(self):
        random.seed(0)
        mu_arms = random.uniform(low = 0.25, high = 0.8, size = 10)
        super().__init__(time_horizon = 1000, mu_arms = mu_arms, num_runs = 100, init_exploration = 5)
        self.optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
        self.second_optimal_arm_index = np.flip(np.argsort(mu_arms))[1]
        self.current_exponential = 1

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.get_K())

        for i in range(self.get_K()):
            rewards[i] = random.binomial(n = 1, p = self.get_mu_arm_i(i))

        # switch up rewards if t is in [2^(i - 1), 2^i] if i is odd
        if ((i%2 == 1) and (t <= (2**self.current_exponential))):
            temp = rewards[self.optimal_arm_index]
            rewards[self.optimal_arm_index] = rewards[self.second_optimal_arm_index]
            rewards[self.second_optimal_arm_index] = temp

        if (t == (2**self.current_exponential)):
            self.current_exponential += 1

        return rewards