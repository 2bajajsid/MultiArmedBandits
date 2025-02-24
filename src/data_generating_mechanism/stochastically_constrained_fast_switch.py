import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Stochastically_Constrained_Fast_Switch(Data_Generating_Mechanism):

    def __init__(self, num_arms = 25, time_horizon = 2000):
        mu_arms = np.random.uniform(low = 0.25, high = 0.75, size = num_arms)
        super().__init__(time_horizon = time_horizon, mu_arms = mu_arms, num_runs = 100, init_exploration = 5)
        self.optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
        self.worst_arm_index = np.argsort(mu_arms)[0]

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.get_K())

        for i in range(self.get_K()):
            rewards[i] = random.binomial(n = 1, p = self.get_mu_arm_i(i))

        # switch up rewards if t is in [2^(i - 1), 2^i] if i is odd
        if (t%3 == 1):
            temp = rewards[self.optimal_arm_index]
            rewards[self.optimal_arm_index] = rewards[self.worst_arm_index]
            rewards[self.worst_arm_index] = temp

        return rewards