import numpy as np
from numpy import random
from e2d.arms.poisson_arm import Poisson_Arm
from e2d.model_class.model import Model
import math

class Poisson_Model_Class(Model):
    def __init__(self, Delta, K=2, scale = 0.5):
        super().__init__()
        self.arms = []
        self.K = K
        d = np.random.randint(self.K)
        for i in range(0, K):
            if (i == d):
                self.arms.append(Poisson_Arm(np.abs(np.random.normal(loc = Delta, scale = 0.05))))
            else:
                self.arms.append(Poisson_Arm(np.abs(np.random.normal(loc = 0, scale = 0.05)))) 
        self.arm_means = np.zeros(shape = self.K)
        for i in range(K):
            self.arm_means[i] = self.arms[i].get_mean()
    
    def get_optimal_arm_index(self):
        return np.argmax(self.arm_means)

    def generate_observation(self):
        o_t = np.zeros(shape = self.K)
        for i in range(self.K):
            o_t[i] = self.arms[i].draw_sample()  
        return o_t