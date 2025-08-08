import numpy as np
from numpy import random
from e2d.arms.arm import Arm
import math

class Student_t_Arm(Arm):
    def __init__(self, mean, sd):
        super().__init__()
        self.f_m = mean
        self.f_sd = sd
        self.nu = 3
        self.constant_multiplier = (sd) * np.sqrt(((self.nu - 2) / (self.nu)))

    def draw_sample(self):
        return (self.constant_multiplier * np.random.standard_t(df = self.nu)) + self.f_m 
    
    def get_mean(self):
        return self.f_m