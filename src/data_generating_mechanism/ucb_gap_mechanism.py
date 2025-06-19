import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class UCB_Gap_Mechanism(Data_Generating_Mechanism):   
    def __init__(self, gap, reward_sd, time_horizon, must_update_statistics = True):
        self.d = 2
        self.mustUpdateStatistics = True
        self.gap = gap
        self.reward_sd = reward_sd
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = self.d), 
                         num_runs = time_horizon, 
                         init_exploration = 1)
        
    def initialize_parameters(self, hyperparameters):
        self.mean_estimates = np.zeros(shape = self.d)
        self.num_times_arm = np.zeros(shape = self.d)
        self.mu_arms = np.zeros(shape = self.d)
        self.delta = hyperparameters['delta']
        # the first arm is optimal
        self.mu_arms[0] = np.random.normal(loc = 0, 
                                               scale = self.reward_sd, 
                                               size = 1)
        for i in range(1, self.d):
            self.mu_arms[i] = np.random.normal(loc = self.gap, 
                                               scale = self.reward_sd, 
                                               size = 1)

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
        if self.num_times_arm[j] < self.get_init_exploration():
            return np.inf
        else:
            return self.mean_estimates[j] + (np.sqrt(2 * np.log(1/self.delta) / (self.num_times_arm[j] + 1)))

    def get_rewards(self, t):
        errors = np.random.normal(scale = self.reward_sd, size = self.d)
        return self.mu_arms + errors