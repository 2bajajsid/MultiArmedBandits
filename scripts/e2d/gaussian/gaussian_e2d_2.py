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
T = 1000 # Time Horizon
K = 5 # Number of Arms
M = 10 # Number of Models
NUM_RUNS = 100

OPTIMALITY_GAP = 0.05
DELTA = 0.2

GAP_STRING = "Very_High"
MC_STRING = "Mid_Hoeff"

meta_algo = Meta_Algo(M = M, K = K, T = T, num_runs = NUM_RUNS,
                      optimality_gap=OPTIMALITY_GAP, delta=DELTA,
                      file_name = "gaussian/Gaussian_{0}_Gap_{1}_MC.".format(GAP_STRING, MC_STRING),
                      title = "Averaged Regret Over Time [Gaussian Model Class ({0} Gap, {1} MC)]".format(GAP_STRING, MC_STRING))
meta_algo.compute_averaged_regret()
