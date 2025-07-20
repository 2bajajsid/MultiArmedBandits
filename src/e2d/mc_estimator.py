import numpy as np
from numpy import random

# MC Estimator
class MC_Estimator():
    def __init__(self, finite_model_class):
        super().__init__()
        self.finite_model_class = finite_model_class
    
    def clear(self):
        self.samples_drawn = np.zeros(shape = (self.finite_model_class.get_model_class_length(),
                                               self.finite_model_class.K,
                                               self.m))
        
    def draw_samples(self, m):
        # (M x K x m)
        self.m = m
        self.samples_drawn = np.zeros(shape = (self.finite_model_class.get_model_class_length(),
                                               self.finite_model_class.K, 
                                               self.m))
        
        for i in range(self.finite_model_class.get_model_class_length()):
            self.samples_drawn[i, :] = self.finite_model_class.draw_sample_from_model_index(i, self.m)

    def get_f_m_hat(self):
        f_m_hat = np.average(self.samples_drawn, axis=2)
        print("Mean Estimates {} with sample size {}".format(f_m_hat, self.m))