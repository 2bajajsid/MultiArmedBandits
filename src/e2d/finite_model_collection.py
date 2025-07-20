from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract class for a finite model collection; 
# Each model class contains references to a finite number of models
class Finite_Model_Collection(ABC):
    @abstractmethod
    def get_model_class_length(self):
        return
    
    @abstractmethod
    def get_ot(self, action_pi):
        return
    
    @abstractmethod
    def compute_instantaneous_regret(self, p_t):
        return
    
    @abstractmethod
    def draw_sample_from_model_index(self):
        return
    
    @abstractmethod
    def print_description(self):
        return