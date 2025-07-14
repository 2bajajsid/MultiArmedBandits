import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from data_generating_mechanism.ucb_gap_mechanism import UCB_Gap_Mechanism
from bandit_algorithms.linear_gaussian_ucb import Linear_Gaussian_UCB
from game.partial_information_game import Partial_Information_Game
from scipy.optimize import Bounds
import numpy as np
import math
np.random.seed(0)

HIGH_GAP = 4.00
MEDIUM_GAP = 2.5
LOW_GAP = 0.75
VERY_LOW_GAP = 0.0025
T = 1000

print("High Gap Experiment")

# High Gap 2-armed experiment
data_job = UCB_Gap_Mechanism(gap = HIGH_GAP, reward_sd=1, time_horizon=T)
ucb_hyperparameters = [{'delta': 1/(T**2)},
                       {'delta': 1/T},
                       {'delta': 0.01},
                       {'delta': 0.05}, 
                       {'delta': 0.1},
                       {'delta': 0.2},
                       {'delta': 0.3},
                       {'delta': 0.4},
                       {'delta': 0.5},
                       {'delta': 0.6}, 
                       {'delta': 0.7},
                       {'delta': 0.725},
                       {'delta': 0.7598},
                       {'delta': 0.775},
                       {'delta': 0.8},
                       {'delta': 0.9},
                       {'delta': 0.95}]
partial_info_ground = Partial_Info_Play_Ground(bandit_algorithms=[Linear_Gaussian_UCB(data_job, "UCB1")],
                                              hyperparameters=[ucb_hyperparameters],
                                              plot_label = "High_Gap",
                                              plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/stat_900_experiment_3/",
                                              gap=HIGH_GAP)
partial_info_ground.plot_regret_as_function_of_hyperparameters(vlines = [0.7598, 0.001])