import numpy as np
from numpy import random
from e2d.arms.arm import Arm
import math

class Poisson_Arm(Arm):
    def __init__(self, lam):
        super().__init__()
        self.f_m = lam

    def draw_sample(self):
        return random.poisson(lam = self.f_m)
    
    def get_mean(self):
        return self.f_m