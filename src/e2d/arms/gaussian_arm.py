import numpy as np
from numpy import random
from e2d.arms.arm import Arm
import math

class Gaussian_Arm(Arm):
    def __init__(self, mean, sd):
        super().__init__()
        self.f_m = mean
        self.f_sd = sd

    def draw_sample(self):
        normal_random = random.normal(loc = self.f_m, 
                             scale = self.f_sd)
        if (normal_random > 1):
            return 1
        elif (normal_random < -1):
            return -1
        else:
            return normal_random
    
    def get_mean(self):
        return self.f_m