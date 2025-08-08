import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class UCB_Gap_Mechanism(Data_Generating_Mechanism):   
    def __init__(self, gap, reward_sd, time_horizon, prior_samples = 3000, prior_repeats = 40, must_update_statistics = True):
        self.d = 2
        self.mustUpdateStatistics = True
        self.reward_sd = reward_sd
        self.gap = gap
        self.prior_samples = prior_samples
        self.prior_repeats = prior_repeats
        self.problems_sampled = np.zeros(shape = (self.prior_samples * self.prior_repeats, self.d))
        self.current_M = (self.prior_repeats * self.prior_samples)
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = self.d), 
                         num_runs = prior_samples * prior_repeats, 
                         init_exploration = 1)
        
    def initialize_parameters(self, hyperparameters):
        if (self.current_M == self.prior_repeats * self.prior_samples):
            self.current_M = 0
            for i in range(self.prior_samples):
                for f in range(1):
                    self.problems_sampled[int(i*self.prior_repeats)][f] = np.random.normal(loc = 0, scale = np.sqrt(0.25), size = 1)
                self.problems_sampled[int(i*self.prior_repeats)][1] = np.random.normal(loc = self.gap, 
                                                        scale = np.sqrt(0.25), 
                                                        size = 1)
                for j in range(1, self.prior_repeats):
                    for f in range(1):
                        self.problems_sampled[int(i*self.prior_repeats + j)][f] = np.random.normal(loc = 0, scale = np.sqrt(0.25), size = 1)
                    self.problems_sampled[int(i*self.prior_repeats + j)][1] = self.problems_sampled[int(i*self.prior_repeats)][1]
        self.mean_estimates = np.zeros(shape = self.d)
        self.num_times_arm = np.zeros(shape = self.d)
        self.mu_arms = np.zeros(shape = self.d)
        self.delta = hyperparameters['delta']
        for f in range(2):
            self.mu_arms[f] = self.problems_sampled[self.current_M][f]
        self.current_M += 1

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
        if self.num_times_arm[j] < 2:
            return np.inf
        else:
            return self.mean_estimates[j] + (np.sqrt(2 * np.log(1/self.delta) / (self.num_times_arm[j])))

    def get_rewards(self, t):
        errors = np.random.normal(scale = self.reward_sd, size = self.d)
        return self.mu_arms + errors