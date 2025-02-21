import numpy as np
from numpy import random
from algorithm_class import Bandit_Algorithm_PI, Bandit_Algorithm_FI
import math

class Gaussian_Thompson_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism, sigma_sq):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.sigma_sq = sigma_sq

        self.__data_generating_mechanism = data_generating_mechanism
        self.__label =  "Gaussian Thompson Sampling (sigma^2: {:0.2f})".format(self.sigma_sq)

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t):
        if (t < self.exploration_phase_length):
            return math.floor(t / self.init_exploration)
        else:
            perturbations = random.normal(scale = math.sqrt(self.sigma_sq * (self.data_generating_mechanism.get_T() - t)), 
                                          size = self.data_generating_mechanism.get_K())
            return np.argmin(np.sum(losses, axis=1) + perturbations)
        
class Gaussian_Thompson_PI(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism, sigma_sq):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.num_bootstrap_simulations = 100
        self.sigma_sq = sigma_sq

        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "Gaussian Thompson Sampling (sigma_sq: {:0.2f})".format(self.sigma_sq)

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label
    
    @property
    def current_sampling_distribution(self):
        return self.__current_sampling_distribution
    
    @current_sampling_distribution.setter
    def current_sampling_distribution(self, distr):
        self.__current_sampling_distribution = distr

    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        if (t < self.exploration_phase_length):
            A_t = math.floor(t / self.data_generating_mechanism.get_init_exploration())
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
        else:
            num_counts = np.zeros(shape = self.K)
            
            for n in range(self.num_bootstrap_simulations):
                perturbations = random.normal(scale = math.sqrt(self.sigma_sq * (self.data_generating_mechanism.get_T() - t)), 
                                          size = self.data_generating_mechanism.get_K())
                arm_chosen_this_simulation = np.argmin(np.sum(importance_weighted_losses, axis = 1) + perturbations)
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1
                if (self.num_bootstrap_simulations == n + 1):
                    A_t = arm_chosen_this_simulation
                
            self.current_sampling_distribution = num_counts / sum(num_counts)
        
        return A_t