from abc import ABC, abstractmethod
import numpy as np
from numpy import random

# Abstract Class 
# for an arm
class Arm(ABC):
    @abstractmethod
    def draw_sample(self):
        pass