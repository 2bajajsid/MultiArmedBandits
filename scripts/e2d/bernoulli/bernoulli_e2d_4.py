import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from e2d.model_collection.bernoulli_model_collection import Bernoulli_Model_Collection
from e2d.technical_tools.mc_estimator import MC_Estimator
from e2d.oracle.exp_weights_oracle import Exp_Weights_Oracle
from e2d.meta_algo import Meta_Algo
from e2d.players.exp3_player import Exp3_Player
from e2d.players.dec_player import DEC_Player

np.random.seed(900)
T = 3000 # Time Horizon
K = 3 # Number of Arms
M = 10 # Number of Models
NUM_RUNS = 500

OPTIMALITY_GAP = 0.4
DELTA = 0.8

model_class = Bernoulli_Model_Collection(K = K, M = M, Optimality_Gap=OPTIMALITY_GAP)
mc_estimator = MC_Estimator(finite_model_class = model_class)
m = mc_estimator.get_m_for_bernoulli_model(delta = DELTA,
                 optimality_gap = OPTIMALITY_GAP)

players = []
players.append(Exp3_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M, K=K, model_class=model_class), label="EXP3"))

gamma = [0.5, 1.0, 10.0]
for g in range(len(gamma)):
    players.append(DEC_Player(M = M, T = T, K = K, numRuns = NUM_RUNS, algEst=Exp_Weights_Oracle(T = T, M = M, K = K, model_class=model_class), 
                              gamma = gamma[g], label="DEC (gamma): {}".format(gamma[g])))

meta_algo = Meta_Algo(M = M, K = K, T = T, 
                      mc_estimator= mc_estimator, players=players,
                      finite_model_class = model_class, num_runs = NUM_RUNS, m = m, 
                      file_name = "bernoulli/Bernoulli_Low_Gap_Low_MC",
                      title = "Averaged Regret Over Time [Bernoulli Model Class (Low Gap, Low MC)]")
meta_algo.compute_averaged_regret()