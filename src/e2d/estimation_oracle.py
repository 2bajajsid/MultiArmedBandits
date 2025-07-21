from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Estimation Oracle class
# is responsible for using some sort of 
# a weighting algorithm and returns the index of 
# the expert that has been most accurate so far
class Estimation_Oracle(ABC):
    @abstractmethod
    def get_m_hat_index(self):
        return
    
    @abstractmethod
    def add_to_training_data_set(self, f_m_hat, r_t):
        return
    
    @abstractmethod
    def current_sampling_distribution(self):
        return
    
    @abstractmethod
    def clear(self):
        return