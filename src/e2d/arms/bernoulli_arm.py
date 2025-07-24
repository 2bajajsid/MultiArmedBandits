import numpy as np
from numpy import random
from e2d.arms.arm import Arm
import math

class Bernoulli_Arm(Arm):
    def __init__(self, mean):
        super().__init__()
        self.f_m = mean

    def draw_sample(self):
        return random.binomial(n = 1, p = self.f_m)