import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from e2d.model_collection.gaussian_model_collection import Gaussian_Model_Collection
from e2d.technical_tools.mc_estimator import MC_Estimator
from e2d.oracle.exp_weights_oracle import Exp_Weights_Oracle
from e2d.meta_algo import Meta_Algo
from e2d.players.exp3_player import Exp3_Player
from e2d.players.dec_player import DEC_Player
from e2d.technical_tools.constants import HELLINGER_SQUARE

np.random.seed(900)
T = 1000 # Time Horizon
K = 10 # Number of Arms
M = 10 # Number of Models
NUM_RUNS = 100

OPTIMALITY_GAP = 0.05
DELTA = 0.2

GAP_STRING = "Low"
MC_STRING = "Low_Sq"
plt.rcParams["figure.figsize"] = (15,6)

model_index_1 = 0
model_index_2 = 8
action = 4

meta_algo = Meta_Algo(M = M, K = K, T = T, num_runs = NUM_RUNS,
                      optimality_gap=OPTIMALITY_GAP, delta=DELTA,
                      file_name = "gaussian/Gaussian_{0}_Gap_{1}_MC.".format(GAP_STRING, MC_STRING),
                      title = "Averaged Regret Over Time [Gaussian Model Class ({0} Gap, {1} MC)]".format(GAP_STRING, MC_STRING),
                      sample_size_type=10,
                      divergence_type=HELLINGER_SQUARE)
monte_carlo = [10, 50, 500, 5000]

derivative_stats = meta_algo.get_true_radon_nikodym_derivatives(dp_or_dq=1, 
                                                                model_index_1=model_index_1, model_index_2=model_index_2, action=action)
plt.plot(derivative_stats[0], derivative_stats[1], label = "True")
for m in monte_carlo:
    stats = meta_algo.get_estimated_radon_nikodym_derivatives(m = m, dp_or_dq=1, 
                                                              model_index_1=model_index_1, model_index_2=model_index_2, action=action)
    plt.plot(stats[0], stats[1], label = "m = {}".format(m))
plt.legend()
plt.title("dq / dr")
plt.xlabel("x")
plt.ylabel("Radon-Nikodym derivative")
plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/gaussian/Gaussian_dq_dr")
plt.close()

derivative_stats = meta_algo.get_true_radon_nikodym_derivatives(dp_or_dq=0, 
                                                                model_index_1=model_index_1, model_index_2=model_index_2, action=action)
plt.plot(derivative_stats[0], derivative_stats[1], label = "True")
for m in monte_carlo:
    stats = meta_algo.get_estimated_radon_nikodym_derivatives(m = m, dp_or_dq=0, 
                                                              model_index_1=model_index_1, model_index_2=model_index_2, action=action)
    plt.plot(stats[0], stats[1], label = "m = {}".format(m))
plt.legend()
plt.title("dp / dr")
plt.xlabel("x")
plt.ylabel("Radon-Nikodym derivative")
plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/gaussian/Gaussian_dp_dr")
