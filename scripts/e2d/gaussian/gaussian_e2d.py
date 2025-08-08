import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from threading import Thread
import multiprocessing as mp
from e2d.meta_algo import Meta_Algo
from e2d.technical_tools.constants import HELLINGER_SQUARE, GAUSSIAN_MODELS

np.random.seed(900)
T = 2500 # Time Horizon
K = 5 # Number of Arms
M = 3 # Number of Models
NUM_RUNS = 50

OPTIMALITY_GAP = 0.5
DELTA = 0.8

GAP_STRING = "Low"
MC_STRING = "V_Low_Sq"
plt.rcParams["figure.figsize"] = (15,6)
plt.switch_backend('Agg') 

def target_func_1(m, label, reg, std, gamma):
    meta_algo = Meta_Algo(M = M, K = K, T = T, num_runs = NUM_RUNS,
                        optimality_gap=OPTIMALITY_GAP, delta=DELTA,
                        file_name = "gaussian/Gaussian_{0}_Gap_{1}_MC.".format(GAP_STRING, MC_STRING),
                        title = "Averaged Regret Over Time [Gaussian Model Class ({0} Gap, {1} MC)]".format(GAP_STRING, MC_STRING),
                        sample_size_type=m,
                        divergence_type=HELLINGER_SQUARE,
                        type_model= GAUSSIAN_MODELS)
    game_winner_stats = meta_algo.compute_averaged_regret()
    for s in range(T):
        reg[s] = game_winner_stats[0][s]
        std[s] = game_winner_stats[1][s]
    gamma.value = game_winner_stats[3]
    
threads = []
sample_sizes = [50, 1, 2, -1]
labels = ['m: 50, ',
          'Subgaussian, ',
          'Asymptotics, ',
          'True Hellinger Square, ']

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    
    empirical_regret = []
    empirical_std = []
    gamma = []

    for m in range(len(sample_sizes)):
        #threads.append(Thread(target=target_func_1, args = (sample_sizes[m], labels[m],)))
        empirical_regret.append(mp.Array('d', range(T)))
        empirical_std.append(mp.Array('d', range(T)))
        gamma.append(mp.Value('d', 0.0))
        threads.append(ctx.Process(target=target_func_1, 
                                   args = (sample_sizes[m], labels[m], 
                                           empirical_regret[m], empirical_std[m], gamma[m])))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

    for i in range(len(sample_sizes)):
        reg = np.zeros(shape = T)
        std = np.zeros(shape = T)
        for t in range(T):
            reg[t] = empirical_regret[i][t] 
            std[t] = empirical_std[i][t]
        plt.errorbar(x = range(T)[0::10],
                y = reg[0::10], 
                yerr = std[0::10], 
                label = labels[i] + "gamma: {}".format(gamma[i].value),
                alpha=0.5)
        
    plt.legend()
    plt.title("Empirical Regret of E2D procedure on Gaussian model classes")
    plt.xlabel("Time Step")
    plt.ylabel("Averaged Regret")
    plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/gaussian/Gaussian_e2d_point_o_eight_neg")
