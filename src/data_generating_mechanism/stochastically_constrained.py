import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Stochastically_Constrained(Data_Generating_Mechanism):

    def __init__(self):
        mu_arms = [0.43693871, 0.36341421, 0.42633424, 0.65485628, 0.51202001, 
                   0.81, 0.33343729, 0.45265664, 0.32908013, 0.26136635]
        super().__init__(time_horizon = 1000, mu_arms = mu_arms, num_runs = 100, init_exploration = 5)
        self.optimal_arm_index = np.flip(np.argsort(mu_arms))[0]
        self.worst_arm_index = np.argsort(mu_arms)[0]
        self.current_exponential = 1

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.get_K())

        for i in range(self.get_K()):
            rewards[i] = random.binomial(n = 1, p = self.get_mu_arm_i(i))

        # switch up rewards if t is in [2^(i - 1), 2^i] if i is odd
        if ((i%2 == 1) and (t <= (2**self.current_exponential))):
            temp = rewards[self.optimal_arm_index]
            rewards[self.optimal_arm_index] = rewards[self.worst_arm_index]
            rewards[self.worst_arm_index] = temp

        if (t == (2**self.current_exponential)):
            self.current_exponential = self.current_exponential + 1

        if (t == self.get_T()):
            self.current_exponential = 1

        return rewards