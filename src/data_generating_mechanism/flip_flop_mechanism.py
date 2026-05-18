import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Flip_Flop_Mechanism(Data_Generating_Mechanism):
        
    def __init__(self, time_horizon = 400, num_arms = 10, M = 100):
        mu_arms = np.random.uniform(low = 0.1, high = 0.5, size = num_arms)
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = mu_arms, 
                         num_runs = M,
                         init_exploration=5)

    def get_rewards(self, t):
        first_half_size = int(self.get_K() / 2)
        second_half_size = self.get_K() - first_half_size
        if t == 0:
            first_half = 1/2 * np.ones(shape = first_half_size) 
            second_half = np.zeros(shape = second_half_size)
            return np.concatenate((first_half, second_half))
        elif t % 2 == 1:
            first_half = np.zeros(shape = first_half_size)
            second_half = np.ones(shape = second_half_size)
            return np.concatenate((first_half, second_half))
        else: 
            first_half = np.ones(shape = first_half_size)
            second_half = np.zeros(shape = second_half_size)
            return np.concatenate((first_half, second_half))
            