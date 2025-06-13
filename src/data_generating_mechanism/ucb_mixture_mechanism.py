import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class UCB_Mixture_Mechanism(Data_Generating_Mechanism):   
    def __init__(self, d = 10, time_horizon = 1000, must_update_statistics = True):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.mustUpdateStatistics = True
        self.d = d
        self.low_gap_mean_vector = np.random.uniform(low = 0, high = 0.25, size = self.d)
        self.high_gap_mean_vector = np.random.uniform(low = 0, high = 2.5, size = self.d)
        self.gamma = 0.3
        print(self.gamma * self.low_gap_mean_vector + (1 - self.gamma) * self.high_gap_mean_vector)
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = d), 
                         num_runs = 1000, 
                         init_exploration = 1)
        
    def initialize_parameters(self, hyperparameter):
        self.mean_estimates = np.zeros(shape = self.d)
        self.num_times_arm = np.ones(shape = self.d)
        self.gamma = 0.3
        self.confidence_width = hyperparameter
        self.mu_arms = self.gamma * np.random.multivariate_normal(self.low_gap_mean_vector, np.eye(self.d), size = 1)[0] 
        self.mu_arms += (1 - self.gamma) * np.random.multivariate_normal(self.high_gap_mean_vector, np.eye(self.d), size = 1)[0]

    def get_optimal_arm_mean(self):
        return np.max(self.mu_arms)
    
    def get_arm_mean(self, idx):
        return self.mu_arms[int(idx)]
        
    def update_statistics(self, arm_index, reward, t):
        t_arm = self.num_times_arm[arm_index]
        self.mean_estimates[arm_index] = ((t_arm * self.mean_estimates[arm_index]) + reward) / (t_arm+1)
        self.num_times_arm[arm_index] += 1
        return 
    
    def get_arm_index(self, j, t):
        if (t == 0):
            return np.inf
        else:
            return self.mean_estimates[j] + (self.confidence_width * np.sqrt(2 * np.log(t + 1) / self.num_times_arm[j]))

    def get_rewards(self, t):
        errors = np.random.normal(size = self.d)
        return self.mu_arms + errors