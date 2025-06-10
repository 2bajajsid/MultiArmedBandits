import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Linear_Gaussian_Stochastic(Data_Generating_Mechanism):   
    def __init__(self, d = 10, num_arms = 25, time_horizon = 1000, must_update_statistics = True, init_exploration = 1, delta = 0.05):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.d = d
        self.delta = delta
        self.mustUpdateStatistics = True
        self.posterior_mean = np.zeros(d)
        self.posterior_covariance = 10 * np.identity(d)
        self.theta_star = np.random.multivariate_normal(self.posterior_mean, 
                                                         self.posterior_covariance, 
                                                         size = 1)[0]
        self.alpha = 1 + np.sqrt(np.log(2 / delta) / 2)
        self.theta_hat = np.zeros(d)
        self.lambda_reg = 0.5
        self.V = self.lambda_reg * np.identity(d)
        self.x = np.zeros(d)
        self.b = np.zeros(d)

        # (num_arms x d) matrix
        self.feature_vectors = np.reshape(np.random.uniform(low = -1/np.sqrt(d), 
                                                high = 1/np.sqrt(d), 
                                                size = num_arms * d),
                                    shape = (num_arms, d))
        self.mu_arms = self.feature_vectors @ self.theta_star

        super().__init__(time_horizon = time_horizon, 
                         mu_arms = self.mu_arms, 
                         num_runs = 50, 
                         init_exploration = init_exploration)
        
    def update_statistics(self, arm_index, reward):
        x = self.get_arm_feature_map(arm_index)
        self.V = self.V + (np.transpose(x) @ x)
        self.b = self.b + (reward * np.transpose(x)) 
        self.theta_hat = np.linalg.inv(self.V) @ self.b
        return 
    
    def get_arm_feature_map(self, j):
        return self.feature_vectors[j, :]
    
    def get_m2(self):
        return np.sqrt(self.d * 3)
    
    def get_sqrt_beta_t(self, t):
        first_part = self.get_m2() * np.sqrt(self.lambda_reg)
        second_part = np.sqrt(2 * np.log(1 / self.lambda_reg) + np.log(np.linalg.det(self.V)/self.lambda_reg**self.d))
        return first_part + second_part
    
    def get_arm_index(self, j, t):
        arm_feature = self.get_arm_feature_map(j)
        return (np.transpose(self.theta_hat) @ arm_feature) + self.get_sqrt_beta_t(t) * (np.sqrt(arm_feature @ np.linalg.inv(self.V) @ np.transpose(arm_feature)))

    def get_rewards(self, t):
        sub_gaussian_error_terms = np.random.normal(size = 25)
        return self.mu_arms + sub_gaussian_error_terms