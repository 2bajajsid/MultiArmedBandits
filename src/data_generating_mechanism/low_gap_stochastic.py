import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Low_Gap_Stochastic(Data_Generating_Mechanism):
        
    def __init__(self):
        mu_arms = [0.68693871, 0.36341421, 0.42633424, 0.55485628, 0.61202001, 
                   0.64762386, 0.33343729, 0.59265664, 0.32908013, 0.47]
        super().__init__(time_horizon = 1000, mu_arms = mu_arms, num_runs = 100, init_exploration = 5)

    def get_rewards(self, t):
        rewards = np.zeros(shape = self.get_K())

        for i in range(self.get_K()):
            rewards[i] = random.binomial(n = 1, p = self.get_mu_arm_i(i))

        return rewards