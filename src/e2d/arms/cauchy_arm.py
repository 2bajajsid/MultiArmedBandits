import numpy as np
from numpy import random
from e2d.arms.arm import Arm
import math

class Cauchy_Arm(Arm):
    def __init__(self, mean):
        super().__init__()
        self.f_m = mean

    def draw_sample(self):
        u = random.normal(loc = 0.0, scale = 1.0)
        v = random.normal(loc = 0.0, scale = 1.0)
        return (u / v) + self.f_m
    
    def get_mean(self):
        return self.f_m