import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism

class Linear_Posterior_Sampling_Stochastic(Data_Generating_Mechanism):   
    def __init__(self, d = 10, posterior_mean = np.zeros(10), posterior_covariance = 10 * np.identity(10),
                num_arms = 100, time_horizon = 1000, 
                must_update_statistics = True, init_exploration = 1, misspecify = False):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.d = d
        self.mustUpdateStatistics = True
        self.data_posterior_mean = np.zeros(10)
        self.data_posterior_covariance =  10 * np.identity(10)
        self.num_arms = num_arms
        self.misspecify = misspecify
       
        # state variables to compute posteriors
        self.posterior_mean = posterior_mean
        self.posterior_covariance = posterior_covariance
        self.precision_matrix = np.zeros(shape = (d, d))
        self.X = np.zeros(shape = (time_horizon, d))
        self.Y = np.zeros(shape = time_horizon)

        self.initialize_parameters()
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = num_arms), 
                         num_runs = 500, 
                         init_exploration = init_exploration)
        
    def initialize_parameters(self):
        if self.misspecify == True:
            self.theta_star = np.random.standard_cauchy(size = 10)
        else:
            self.theta_star = np.random.multivariate_normal(self.data_posterior_mean, 
                                                            self.data_posterior_covariance, 
                                                            size = 1)[0]
         
        # (num_arms x d) matrix
        self.feature_vectors = np.reshape(np.random.uniform(low = -1/np.sqrt(self.d), 
                                                high = 1/np.sqrt(self.d), 
                                                size = self.num_arms * self.d),
                                    shape = (self.num_arms, self.d))
        self.mu_arms = self.feature_vectors @ self.theta_star
        self.theta_hat = np.random.multivariate_normal(self.posterior_mean, self.posterior_covariance, size = 1)[0]
        
    def update_statistics(self, arm_index, reward, t):
        x = self.get_arm_feature_map(arm_index)
        self.X[t, :] = x
        self.Y[t] = reward

        X_t = self.X[:t, ]
        Y_t = self.Y[:t]

        # because the posterior also has a 
        # multi-variate normal distribution
        self.precision_matrix = np.matmul(np.transpose(X_t), X_t) + (1/10) * np.identity(self.d)
        self.posterior_covariance = np.linalg.inv(self.precision_matrix)
        self.posterior_mean = np.matmul(self.posterior_covariance, np.matmul(np.transpose(X_t), Y_t))
        self.theta_hat = np.random.multivariate_normal(self.posterior_mean, self.posterior_covariance, size = 1)[0]

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
    
    def get_arm_index(self, j, t):
        arm_feature = self.get_arm_feature_map(j)
        return np.dot(self.theta_hat, arm_feature)

    def get_rewards(self, t):
        sub_gaussian_error_terms = np.random.normal(size = self.num_arms)
        return self.mu_arms + sub_gaussian_error_terms