from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract Class for a Full-Information Bandit Algorithm
class Bandit_Algorithm_FI:

    @abstractmethod
    def get_arm_to_pull(self, losses, t):
        pass

    @property
    @abstractmethod
    def data_generating_mechanism(self):
        pass

    @property
    @abstractmethod
    def label(self):
        pass
        
# Abstract Class for a Partial-Information Bandit Algorithm
class Bandit_Algorithm_PI:

    @abstractmethod
    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        pass

    @property
    @abstractmethod
    def current_sampling_distribution(self):
        pass

    @property
    @abstractmethod
    def data_generating_mechanism(self):
        pass

    @property
    @abstractmethod
    def label(self):
        pass