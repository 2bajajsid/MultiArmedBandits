import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Linear_Gaussian_Stochastic(Data_Generating_Mechanism):   
    def __init__(self, post, d = 10, lambda_reg = 0.25, num_arms = 100, time_horizon = 1000, must_update_statistics = True, init_exploration = 1, delta = 0.01):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.d = d
        self.delta = delta
        self.mustUpdateStatistics = True
        self.posterior_mean = np.zeros(d)
        self.posterior_covariance = post * np.identity(d)
        self.num_arms = num_arms
       
        self.theta_hat = np.zeros(d)
        self.lambda_reg = lambda_reg
        self.V = self.lambda_reg * np.identity(d)
        self.b = np.zeros(d)

        self.initialize_parameters()

        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = num_arms), 
                         num_runs = 25, 
                         init_exploration = init_exploration)
        
    def initialize_parameters(self):
        self.theta_star = np.random.multivariate_normal(self.posterior_mean, 
                                                         self.posterior_covariance, 
                                                         size = 1)[0]
         
        # (num_arms x d) matrix
        self.feature_vectors = np.reshape(np.random.uniform(low = -1/np.sqrt(self.d), 
                                                high = 1/np.sqrt(self.d), 
                                                size = self.num_arms * self.d),
                                    shape = (self.num_arms, self.d))
        self.mu_arms = self.feature_vectors @ self.theta_star
        
    def update_statistics(self, arm_index, reward, t):
        x = self.get_arm_feature_map(arm_index)
        self.V = self.V + np.outer(x, x)
        self.b = self.b + (reward * x)
        self.theta_hat = np.linalg.inv(self.V) @ self.b
        return 

    def get_arm_mean(self, j):
        arm_feature = self.get_arm_feature_map(j)
        return np.dot(self.theta_star, arm_feature)
    
    def get_optimal_arm_mean(self):
        arm_feature = self.get_arm_feature_map(np.argmax(self.mu_arms))
        return np.dot(self.theta_star, arm_feature)
    
    def get_optimal_arm_index(self):
        return np.argmax(self.mu_arms)

    def get_arm_feature_map(self, j):
        return self.feature_vectors[int(j), :]
    
    def get_m2(self):
        return np.sqrt(self.d * 5)
    
    def get_beta(self, t):
        first_part = np.sqrt(self.lambda_reg) * self.get_m2()
        second_part = np.sqrt(2 * np.log(1 / self.delta) + np.log(np.linalg.det(self.V) / self.lambda_reg**self.d))
        return first_part + second_part
    
    def compareWith(self, i, t):
        optimal_arm_idx = self.get_optimal_arm_index()
        chosen_true = self.get_arm_mean(i)
        chosen_index = self.get_arm_index(i, t)
        optimal_true = self.get_arm_mean(optimal_arm_idx)
        optimal_index = self.get_arm_index(optimal_arm_idx, t)
        print("Chosen - True mean {} Index {}".format(chosen_true, chosen_index))
        print("Optimal - True mean {} Index {}".format(optimal_true, optimal_index))
        print("**")
        return 
    
    def get_arm_index(self, j, t):
        arm_feature = self.get_arm_feature_map(j)
        return np.dot(self.theta_hat, arm_feature) + self.get_beta(t) * (np.sqrt(arm_feature @ np.linalg.inv(self.V) @ np.transpose(arm_feature)))

    def get_rewards(self, t):
        sub_gaussian_error_terms = np.random.normal(size = self.num_arms)
        return self.mu_arms + sub_gaussian_error_terms