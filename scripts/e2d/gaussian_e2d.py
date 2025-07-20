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

mc_estimator.draw_samples(m = 25)
f_m_hat = mc_estimator.get_f_m_hat()

mc_estimator.clear()
mc_estimator.draw_samples(m = 100)
f_m_hat = mc_estimator.get_f_m_hat()
