from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract Class 
# for a player
class Player(ABC):
    @abstractmethod
    def select_action(self):
        pass

    @abstractmethod
    def update_training_dataset(self):
        pass

    @abstractmethod
    def plot_averaged_regret(self):
        pass