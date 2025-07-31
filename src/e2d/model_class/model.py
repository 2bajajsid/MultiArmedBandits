from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract class for a model; 
# Each model will contain 
# references to k arms
class Model(ABC):
    @abstractmethod
    def get_optimal_arm_index(self):
        pass

    @abstractmethod
    def generate_observation(self):
        pass

    def get_delta(self):
        arms_sorted = np.abs(np.diff(self.arm_means))
        return np.min(arms_sorted)