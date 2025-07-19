from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract class for a finite model collection; 
# Each model class contains references to a finite number of models
class Finite_Model_Collection(ABC):
    @abstractmethod
    def get_model_class_length(self):
        return