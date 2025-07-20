import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from e2d.gaussian_model_collection import Gaussian_Model_Collection
from e2d.mc_estimator import MC_Estimator

model_Class = Gaussian_Model_Collection(K = 2, M = 5)
model_Class.print_description()

mc_estimator = MC_Estimator(finite_model_class = model_Class)

mc_estimator.draw_samples(m = 100000)
f_m_hat = mc_estimator.get_f_m_hat()

mc_estimator.print_true_squared_hellinger_distance()
mc_estimator.get_sq_hellinger_divergence_hat()
