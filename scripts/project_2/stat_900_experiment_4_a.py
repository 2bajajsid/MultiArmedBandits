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

print("Medium Gap Experiment")

# High Gap 2-armed experiment
data_job = UCB_Gap_Mechanism(gap = MEDIUM_GAP, time_horizon=T)
ucb_hyperparameters = [{'reward_sd': 2.0, 'delta': 0.14939},
                      {'reward_sd': 1.75, 'delta': 0.14939},
                       {'reward_sd': 1.5, 'delta': 0.14939},
                       {'reward_sd': 1.25, 'delta': 0.14939}, 
                       {'reward_sd': 1, 'delta': 0.14939},
                       {'reward_sd': 0.5, 'delta': 0.14939}, 
                       {'reward_sd': 0.25, 'delta': 0.14939},
                       {'reward_sd': 0.1, 'delta': 0.14939},
                       {'reward_sd': 2.0, 'delta': 0.1277},
                      {'reward_sd': 1.75, 'delta': 0.1277},
                       {'reward_sd': 1.5, 'delta': 0.1277},
                       {'reward_sd': 1.25, 'delta': 0.1277}, 
                       {'reward_sd': 1, 'delta': 0.1277},
                       {'reward_sd': 0.5, 'delta': 0.1277}, 
                       {'reward_sd': 0.25, 'delta': 0.1277},
                       {'reward_sd': 0.1, 'delta': 0.1277},
                       {'reward_sd': 2.0, 'delta': 0.001},
                      {'reward_sd': 1.75, 'delta': 0.001},
                       {'reward_sd': 1.5, 'delta': 0.001},
                       {'reward_sd': 1.25, 'delta': 0.001}, 
                       {'reward_sd': 1, 'delta': 0.001},
                       {'reward_sd': 0.5, 'delta': 0.001}, 
                       {'reward_sd': 0.25, 'delta': 0.001},
                       {'reward_sd': 0.1, 'delta': 0.001}]
partial_info_ground = Partial_Info_Play_Ground(bandit_algorithms=[Linear_Gaussian_UCB(data_job, "UCB1")],
                                              hyperparameters=[ucb_hyperparameters],
                                              plot_label = "Medium_Gap",
                                              plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/stat_900_exp_4/",
                                              gap=MEDIUM_GAP)
partial_info_ground.plot_regret_as_function_of_hyperparameters_2(vlines=[1], del_vals = [0.14939, 0.1277, 0.001])