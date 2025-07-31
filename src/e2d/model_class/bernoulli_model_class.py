import numpy as np
from numpy import random
from e2d.arms.bernoulli_arm import Bernoulli_Arm
from e2d.model_class.model import Model
import math

class Bernoulli_Model_Class(Model):
    def __init__(self, Delta, K=2, scale = 0.5):
        super().__init__()
        self.arms = []
        self.K = K
        d = np.random.randint(self.K)
        for i in range(0, K):
            if (i == d):
                mean = np.abs(np.random.normal(loc = Delta, scale = 0.01))
                mean = (mean if mean < 1.0 else 0.99)
                self.arms.append(Bernoulli_Arm(mean))
            else:
                mean = np.abs(np.random.normal(loc = 0, scale = 0.01))
                mean = (mean if mean < 1.0 else 0.99)
                self.arms.append(Bernoulli_Arm(mean))
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