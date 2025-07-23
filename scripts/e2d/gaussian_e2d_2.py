import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from e2d.gaussian_model_collection import Gaussian_Model_Collection
from e2d.mc_estimator import MC_Estimator
from e2d.exp_weights_oracle import Exp_Weights_Oracle
from e2d.meta_algo import Meta_Algo
from e2d.exp3_player import Exp3_Player
from e2d.dec_player import DEC_Player

np.random.seed(900)
T = 500 # Time Horizon
K = 10 # Number of Arms
M = 25 # Number of Models
NUM_RUNS = 1000
m = 1000 # Monte-Carlo Sample Size

model_class = Gaussian_Model_Collection(K = K, M = M, Optimality_Gap=2.5)
mc_estimator = MC_Estimator(finite_model_class = model_class)
exp3_player = Exp3_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M), label="EXP3: ")
dec_1_player = DEC_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M), gamma = np.sqrt(T), label="DEC (gamma): sqrt(T)")
dec_2_player = DEC_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M), gamma = T, label="DEC (gamma): T")
dec_3_player = DEC_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M), gamma = T**2, label="DEC (gamma): T^2")
dec_4_player = DEC_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M), gamma = T**4, label="DEC (gamma): T^4")
meta_algo = Meta_Algo(M = M, K = K, T = T, 
                      mc_estimator= mc_estimator, players=[exp3_player, dec_1_player, dec_2_player, dec_3_player, dec_4_player],
                      finite_model_class = model_class, num_runs = NUM_RUNS, m = m, 
                      file_name = "Gaussian_High_Gap_High_MC",
                      title = "Averaged Regret Over Time [Gaussian Model Class (High Gap, High MC)]")
meta_algo.compute_averaged_regret()