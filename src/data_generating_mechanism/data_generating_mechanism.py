import numpy as np
from numpy import random
from abc import ABC, abstractmethod

# Abstract Class for a full-information Bandit Algorithm
class Data_Generating_Mechanism(ABC):
    def __init__(self, mu_arms, time_horizon, num_runs, init_exploration):
        self.__num_arms = len(mu_arms)
        self.__mu_arms = mu_arms
        self.__time_horizon = time_horizon
        self.__num_runs = num_runs
        self.__initial_exploration_per_arm = init_exploration

    def get_K(self):
        return self.__num_arms

    def get_M(self):
        return self.__num_runs
    
    def get_T(self):
        return self.__time_horizon
    
    def get_exploration_phase_length(self):
        return self.__initial_exploration_per_arm * self.get_K()
    
    def get_init_exploration(self):
        return self.__initial_exploration_per_arm
    
    def get_mu_arm_i(self, i):
        return self.__mu_arms[i]

    @abstractmethod
    def get_rewards(self, t):
        pass
    
