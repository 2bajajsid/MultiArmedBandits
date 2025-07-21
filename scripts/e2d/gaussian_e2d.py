import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from e2d.gaussian_model_collection import Gaussian_Model_Collection
from e2d.mc_estimator import MC_Estimator
from e2d.exp_weights_oracle import Exp_Weights_Oracle
from e2d.meta_algo import Meta_Algo

np.random.seed(900)
T = 10000 # Time Horizon
K = 3 # Number of Arms
M = 25 # Number of Models
NUM_RUNS = 2000
m = 1000 # Monte-Carlo Sample Size

model_class = Gaussian_Model_Collection(K = K, M = M)
mc_estimator = MC_Estimator(finite_model_class = model_class)
exp3_oracle = Exp_Weights_Oracle(T = T, M = M)
meta_algo = Meta_Algo(M = M, K = K, T = T, 
                      mc_estimator= mc_estimator, alg_est = exp3_oracle, 
                      finite_model_class = model_class, num_runs = NUM_RUNS, m = m)
meta_algo.compute_averaged_regret()
