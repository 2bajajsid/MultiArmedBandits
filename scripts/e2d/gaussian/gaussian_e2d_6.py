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

np.random.seed(900)
T = 250 # Time Horizon
K = 10 # Number of Arms
M = 5 # Number of Models
NUM_RUNS = 100

OPTIMALITY_GAP = 2.5
DELTA = 0.01

GAP_STRING = "Low"
MC_STRING = "High"

meta_algo = Meta_Algo(M = M, K = K, T = T, num_runs = NUM_RUNS,
                      optimality_gap=OPTIMALITY_GAP, delta=DELTA,
                      file_name = "gaussian/Gaussian_{0}_Gap_{1}_MC.".format(GAP_STRING, MC_STRING),
                      title = "Averaged Regret Over Time [Gaussian Model Class ({0} Gap, {1} MC)]".format(GAP_STRING, MC_STRING))
meta_algo.compute_averaged_regret()