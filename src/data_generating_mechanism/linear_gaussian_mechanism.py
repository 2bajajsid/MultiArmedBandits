import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Linear_Gaussian_Stochastic(Data_Generating_Mechanism):   
    def __init__(self, d = 10, true = 5, num_arms = 100, time_horizon = 1000, prior_samples = 200, prior_repeats = 30, must_update_statistics = True, init_exploration = 1):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.d = d
        self.mustUpdateStatistics = True
        self.posterior_mean = np.zeros(d)
        self.true = true
        self.posterior_covariance = self.true * np.identity(d)
        self.num_arms = num_arms
        self.time_horizon = time_horizon
        self.prior_samples = prior_samples
        self.prior_repeats = prior_repeats
        self.theta_star_sampled = np.zeros(shape = (self.prior_samples * self.prior_repeats, self.d))
        self.feature_vectors_sampled = np.zeros(shape = (self.prior_samples * self.prior_repeats, self.num_arms, self.d))
        self.current_M = (self.prior_repeats * self.prior_samples)

        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = num_arms), 
                         num_runs = prior_samples * prior_repeats, 
                         init_exploration = init_exploration)
        
    def initialize_parameters(self, hyperparameters):
        self.lambda_reg = hyperparameters['lambda']
        self.delta = 1 / self.time_horizon
        self.V = self.lambda_reg * np.identity(10)
        self.theta_hat = np.zeros(self.d)
        self.b = np.zeros(self.d)
        if (self.current_M == self.prior_repeats * self.prior_samples):
            self.current_M = 0
            for i in range(self.prior_samples):
                self.theta_star_sampled[int(i*self.prior_repeats), :] = np.random.normal(loc = self.posterior_mean[0], scale = np.sqrt(self.posterior_covariance[0][0]), size = 10)
                self.feature_vectors_sampled[int(i*self.prior_repeats), :, :] = np.reshape(np.random.uniform(low = -1/np.sqrt(self.d), high = 1/np.sqrt(self.d), size = self.num_arms * self.d), shape = (self.num_arms, self.d))
                for j in range(1, self.prior_repeats):
                    self.theta_star_sampled[int(i*self.prior_repeats) + j, :] = self.theta_star_sampled[int(i*self.prior_repeats), :]
                    self.feature_vectors_sampled[int(i*self.prior_repeats) + j, :, :] = self.feature_vectors_sampled[int(i*self.prior_repeats), :]
        # (num_arms x d) - matrix
        self.mu_arms = self.feature_vectors_sampled[self.current_M, :, :] @ self.theta_star_sampled[self.current_M, :]
        self.current_M += 1
        
    def update_statistics(self, arm_index, reward, t):
        x = self.get_arm_feature_map(arm_index)
        self.V = self.V + np.outer(x, x)
        self.b = self.b + (reward * x)
        self.theta_hat = np.linalg.inv(self.V) @ self.b
        return 

    def get_arm_mean(self, j):
        arm_feature = self.get_arm_feature_map(j)
        return np.dot(self.theta_star_sampled[self.current_M-1, :], arm_feature)
    
    def get_optimal_arm_mean(self):
        arm_feature = self.get_arm_feature_map(np.argmax(self.mu_arms))
        return np.dot(self.theta_star_sampled[self.current_M-1, :], arm_feature)
    
    def get_optimal_arm_index(self):
        return np.argmax(self.mu_arms)

    def get_arm_feature_map(self, j):
        return self.feature_vectors_sampled[self.current_M-1, int(j), :]
    
    def get_m2(self):
        return np.linalg.norm(self.theta_star_sampled[self.current_M-1,:])
    
    def get_beta(self, t):
        first_part = np.sqrt(self.lambda_reg) * self.get_m2()
        second_part = np.sqrt(2 * np.log(1 / self.delta) + np.log(np.linalg.det(self.V) / self.lambda_reg**self.d))
        return first_part + second_part
    
    def get_arm_index(self, j, t):
        arm_feature = self.get_arm_feature_map(j)
        return np.dot(self.theta_hat, arm_feature) + self.get_beta(t) * (np.sqrt(arm_feature @ np.linalg.inv(self.V) @ np.transpose(arm_feature)))

    def get_rewards(self, t):
        sub_gaussian_error_terms = np.random.normal(size = self.num_arms)
        return self.mu_arms + sub_gaussian_error_terms