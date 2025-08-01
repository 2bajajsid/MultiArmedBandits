import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from threading import Thread
import multiprocessing as mp
from e2d.meta_algo import Meta_Algo
from e2d.technical_tools.constants import HELLINGER_SQUARE

np.random.seed(900)
T = 1000 # Time Horizon
K = 10 # Number of Arms
M = 10 # Number of Models
NUM_RUNS = 25

OPTIMALITY_GAP = 0.025
DELTA = 0.8

GAP_STRING = "Low"
MC_STRING = "V_Low_Sq"
plt.rcParams["figure.figsize"] = (15,6)
plt.switch_backend('Agg') 

def target_func_1(m, label):
    meta_algo = Meta_Algo(M = M, K = K, T = T, num_runs = NUM_RUNS,
                        optimality_gap=OPTIMALITY_GAP, delta=DELTA,
                        file_name = "gaussian/Gaussian_{0}_Gap_{1}_MC.".format(GAP_STRING, MC_STRING),
                        title = "Averaged Regret Over Time [Gaussian Model Class ({0} Gap, {1} MC)]".format(GAP_STRING, MC_STRING),
                        sample_size_type=m,
                        divergence_type=HELLINGER_SQUARE)
    game_winner_stats = meta_algo.compute_averaged_regret()

    plt.errorbar(x = range(T),
                y = game_winner_stats[0], 
                yerr = game_winner_stats[1], 
                label = label + game_winner_stats[2])
    
threads = []
sample_sizes = [-1, 10, 50, 250]
labels = ['True Hellinger Square, ', 
          'm: 10, ', 
          'm: 50, ', 
          'm: 250, ']

for m in range(len(sample_sizes)):
    threads.append(Thread(target=target_func_1, args = (sample_sizes[m], labels[m],)))

for t in threads:
    t.start()

for t in threads:
    t.join()

plt.title("Empirical Regret of E2D procedure on Gaussian model classes")
plt.xlabel("Time Step")
plt.ylabel("Averaged Regret")
plt.legend()
plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/gaussian/Gaussian_e2d")
