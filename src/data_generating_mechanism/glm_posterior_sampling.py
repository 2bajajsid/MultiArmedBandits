import numpy as np
from numpy import random
from data_generating_mechanism.data_generating_mechanism import Data_Generating_Mechanism
from scipy import optimize

class GLM_Posterior_Stochastic(Data_Generating_Mechanism):   
    def __init__(self, link, fit_glm, d = 10, num_arms = 100, time_horizon = 1000, must_update_statistics = True, init_exploration = 1):
        # the prior mean and the covariance vector
        # will be posteriors at the first time-step
        self.d = d
        self.mustUpdateStatistics = True
        self.posterior_mean = np.zeros(d)
        self.posterior_covariance = 10 * np.identity(d)
        self.num_arms = num_arms
        self.link_func = link
        #self.reward_gen = reward_gen
        #self.link_func_prime = link_prime
        self.fit_glm = fit_glm
        super().__init__(time_horizon = time_horizon, 
                         mu_arms = np.zeros(shape = num_arms), 
                         num_runs = 100, 
                         init_exploration = init_exploration)
        
    def initialize_parameters(self, hyperparameters):
        self.alpha = hyperparameters['alpha']
        self.r = np.zeros(shape = self.get_T())
        self.m = np.zeros(shape = (self.get_T(), self.d))
        self.M = np.zeros(shape = (self.d, self.d))
        self.curr_time_stamp = 0
        self.theta_star = np.random.normal(loc = self.posterior_mean[0], scale = np.sqrt(self.posterior_covariance[0][0]), size = 10)
        # (num_arms x d) matrix
        self.feature_vectors = np.reshape(np.random.uniform(low = -1/np.sqrt(self.d), 
                                                high = 1/np.sqrt(self.d), 
                                                size = self.num_arms * self.d),
                                    shape = (self.num_arms, self.d))
        self.mu_arms = self.feature_vectors @ self.theta_star 
        for i in range(self.num_arms):
            self.mu_arms[i] = self.link_func(self.mu_arms[i])
        self.theta_hat = np.zeros(self.d)
    
    def estimating_equation(self, theta):
        r_t = self.r[:self.curr_time_stamp]
        m_t = self.m[:self.curr_time_stamp, :]
        sum = 0
        for i in range(self.curr_time_stamp):
            sum += (r_t[i] - self.link_func(m_t[i,:] @ theta)) * np.transpose(m_t[i, :])
        return sum
    
    def jac(self, theta):
        m_t = self.m[:self.curr_time_stamp, :]
        sum = 0
        for i in range(self.curr_time_stamp):
            sum += (self.link_func_prime(m_t[i,:] @ theta)) * (np.outer(m_t[i,:], m_t[i,:]))
        return sum
        
    def update_statistics(self, arm_index, reward, t):
        self.curr_time_stamp += 1
        x = self.get_arm_feature_map(arm_index)
        self.r[t] = reward
        self.m[t, :] = x
        self.M = self.M + np.outer(x, x)
        self.theta_hat = self.fit_glm(x, reward).coef_
        if t % 10 == 0:
            print(self.conf_width)
            print(t)
            print(self.theta_hat)
            print(self.theta_star)
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
    
    def get_arm_index(self, j, t):
        arm_feature = self.get_arm_feature_map(j)
        return self.link_func(np.dot(self.theta_hat, arm_feature)) + self.conf_width * (np.sqrt(arm_feature @ np.linalg.inv(self.M) @ np.transpose(arm_feature)))

    def get_rewards(self, t):
        return self.reward_gen(p = self.mu_arms)