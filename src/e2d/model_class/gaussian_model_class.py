import numpy as np
from numpy import random
from e2d.arms.gaussian_arm import Gaussian_Arm
from e2d.model_class.model import Model
import math

class Gaussian_Model_Class(Model):
    def __init__(self, Delta = -1, K=2, scale = 1.0, arm_means = []):
        super().__init__()
        self.arms = []
        self.K = K
        if (len(arm_means) == 0):
            d = np.random.randint(self.K)
            for i in range(0, K):
                self.arms.append(Gaussian_Arm(np.random.normal(loc = 0,
                                                        scale = 0.01) + (2 * i * Delta),
                                                        sd=0.5))
            temp_arm = self.arms[d]
            self.arms[d] = self.arms[K - 1]
            self.arms[K - 1] = temp_arm       
        else:
            d = np.argmax(arm_means)
            for i in range(K):
                self.arms.append(Gaussian_Arm(arm_means[i] + Delta, 
                                              sd=0.5))
            
            if (d > 0):
                temp_1 = self.arms[d]
                self.arms[d] = self.arms[d-1]
                self.arms[d-1] = temp_1
            else:
                temp_1 = self.arms[d]
                self.arms[d] = self.arms[d+1]
                self.arms[d+1] = temp_1

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