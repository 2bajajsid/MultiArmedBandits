import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Posterior_Sampling_Stochastic(Data_Generating_Mechanism):   
    def __init__(self, d = 10, time_horizon = 1000, 
                must_update_statistics = True, init_exploration = 1, misspecify = False):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.mustUpdateStatistics = True
        self.d = d
        self.low_gap_mean_vector = np.random.uniform(low = 0, high = 0.25, size = self.d)
        self.high_gap_mean_vector = np.random.uniform(low = 0, high = 5, size = self.d)
        self.gamma = 0.25

        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = self.d), 
                         num_runs = 100, 
                         init_exploration = init_exploration)
        
    def initialize_parameters(self, hyperparameters):
        # state variables to compute posteriors
        gamma_hat = hyperparameters['gamma']
        self.mu_arms = np.zeros(shape = self.d)
        self.prior_mean = np.zeros(shape = self.d)
        self.theta_star = np.zeros(shape = self.d)
        self.posterior_mean = np.zeros(shape = self.d)
        for i in range(self.d):
            self.mu_arms[i] = self.gamma * np.random.normal(loc = self.low_gap_mean_vector[i], scale = 1, size = 1)
            self.mu_arms[i] += (1 - self.gamma) * np.random.normal(loc = self.high_gap_mean_vector[i], scale = 1, size = 1)
            self.prior_mean[i] = gamma_hat * np.random.normal(loc = self.low_gap_mean_vector[i], scale = 1, size = 1)
            self.prior_mean[i] += (1 - gamma_hat) * np.random.normal(loc = self.high_gap_mean_vector[i], scale = 1, size = 1)
            self.theta_star[i] = np.random.normal(self.posterior_mean[i], scale = 1, size = 1)
        self.num_times_arm = np.zeros(shape = self.d)
        self.posterior_sd = np.zeros(shape = self.d)
        self.stored_vals = np.zeros(shape = (self.d, self.get_T()))
        
    def update_statistics(self, arm_index, reward, t):
        arm_index = int(arm_index)
        num_times_arm_visited = int(self.num_times_arm[arm_index]) 
        self.stored_vals[arm_index][num_times_arm_visited] = reward
        self.num_times_arm[arm_index] += 1
        x_bar = np.sum(self.stored_vals[arm_index, :num_times_arm_visited]) / self.num_times_arm[arm_index]
        self.posterior_mean[arm_index] = (self.prior_mean[arm_index] + (self.num_times_arm[arm_index] * x_bar)) / (1 + self.num_times_arm[arm_index])
        self.posterior_sd[arm_index] = np.sqrt(1 / (1 + (self.num_times_arm[arm_index] + 1)))
        for i in range(self.d):
            self.theta_star[i] = np.random.normal(self.posterior_mean[i], scale = self.posterior_sd[i], size = 1)

    def get_arm_mean(self, j):
        return self.theta_star[int(j)]
    
    def get_optimal_arm_mean(self):
        return np.max(self.theta_star)
    
    def get_optimal_arm_index(self):
        return np.argmax(self.theta_star)
    
    def get_arm_index(self, j, t):
        return self.theta_star[int(j)]

    def get_rewards(self, t):
        sub_gaussian_error_terms = np.random.normal(size = self.d)
        return self.mu_arms + sub_gaussian_error_terms