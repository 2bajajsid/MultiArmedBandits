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

HIGH_GAP = 2.00
MEDIUM_GAP = 0.5
LOW_GAP = 0.05
VERY_LOW_GAP = 0.005
T = 1000

print("Low Gap Experiment")

# High Gap 2-armed experiment
data_job = UCB_Gap_Mechanism(gap = LOW_GAP, reward_sd=1, time_horizon=T)
ucb_hyperparameters = [{'delta': 1/(T**2)}, 
                       {'delta': 1/T},
                       {'delta': 10/T}, 
                       {'delta': 100/T},
                       {'delta': 1000/T}]
partial_info_ground = Partial_Info_Play_Ground(bandit_algorithms=[Linear_Gaussian_UCB(data_job, "UCB1")],
                                              hyperparameters=[ucb_hyperparameters],
                                              plot_label = "Linear-Gaussian-UCB",
                                              plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/stat_900_high/",
                                              gap=HIGH_GAP)
partial_info_ground.games[0].find_minimum()