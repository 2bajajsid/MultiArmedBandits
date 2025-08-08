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
from e2d.technical_tools.constants import HELLINGER_SQUARE, MEAN_SQUARE
from matplotlib.lines import Line2D

np.random.seed(900)
T = 1000 # Time Horizon
K = 5 # Number of Arms
M = 3 # Number of Models
NUM_RUNS = 100

OPTIMALITY_GAP = 0.025
DELTA = 0.2

GAP_STRING = "Low"
MC_STRING = "Low_Sq"

meta_algo = Meta_Algo(M = M, K = K, T = T, num_runs = NUM_RUNS,
                      optimality_gap=OPTIMALITY_GAP, delta=DELTA,
                      file_name = "gaussian/Gaussian_{0}_Gap_{1}_MC.".format(GAP_STRING, MC_STRING),
                      title = "Averaged Regret Over Time [Gaussian Model Class ({0} Gap, {1} MC)]".format(GAP_STRING, MC_STRING),
                      sample_size_type=10,
                      divergence_type=HELLINGER_SQUARE)
mc_estimator = meta_algo.mc_estimator

fig,ax = plt.subplots(figsize=(15, 6))
monte_carlo = [10, 50, 250, 100000]
print(" : ")
print("Hellinger Square: ")
bias = []
mean_bias_list = []
label = []
for m in monte_carlo:
    mc_estimator.m = m
    stats = mc_estimator.get_bias_of_divergence_hat(divergence_type = HELLINGER_SQUARE)
    mean_bias = stats[0]
    mean_true_divergence = stats[1]
    std_bias = stats[2]
    print("Monte Carlo sample size m {} has mean bias {} with std {} while mean true divergence is {}".format(m, mean_bias, std_bias, mean_true_divergence))
    bias.append(stats[3])
    mean_bias_list.append(stats[0])
    label.append("m: {}".format(m))

y,x, _ = ax.hist(bias, bins=50, histtype='step', density = True, linewidth=2,
        alpha=0.7, label=label, 
        color=['b', 'g', 'r', 'y'])
ax.vlines(mean_bias_list, ymin = 0, ymax = y.max(), 
          color=['b', 'g', 'r', 'y'], 
          linestyles='dotted')

# Edit legend to get lines as legend keys instead of the default polygons
# and sort the legend entries in alphanumeric order
handles, labels = ax.get_legend_handles_labels()
leg_entries = {}
for h, label in zip(handles, labels):
    leg_entries[label] = Line2D([0], [0], color=h.get_facecolor()[:-1],
                                alpha=h.get_alpha(), lw=h.get_linewidth())
labels_sorted, lines = zip(*sorted(leg_entries.items()))
ax.legend(lines, labels_sorted, frameon=False)

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Add annotations
plt.ylabel('Density', labelpad=15)
plt.title('Empirical bias of Squared-Hellinger estimate for Gaussian model classes vs. Monte carlo sample size m', fontsize=14, pad=20)
plt.legend()
plt.xlabel("Empirical bias of Squared-Hellinger estimate")
plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/gaussian/gaussian_bias")
plt.close()