import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Bernoulli_Adversarial(Data_Generating_Mechanism):
        
    def __init__(self, time_horizon = 2000, num_arms = 8, M = 100):
        mu_arms = (1/2) * np.ones(shape = num_arms)
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = mu_arms, 
                         num_runs = M,
                         init_exploration=5)

    def get_rewards(self, t):
        rng = np.random.default_rng()
        return rng.binomial(n=1, 
                            p=1/2, 
                            size=self.get_K())
            