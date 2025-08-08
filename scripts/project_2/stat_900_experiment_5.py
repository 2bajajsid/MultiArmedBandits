import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from data_generating_mechanism.linear_gaussian_mechanism import Linear_Gaussian_Stochastic
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

print("Linear Gaussian UCB Experiment")

# High Gap 2-armed experiment
data_job = Linear_Gaussian_Stochastic(true = 20)
ucb_hyperparameters = [{'lambda': 0.01}, 
                       {'lambda': 0.05},
                       {'lambda': 0.1}, 
                       {'lambda': 0.25},
                       {'lambda': 0.5},
                       {'lambda': 1.0},
                       {'lambda': 1.5},
                       {'lambda': 2.0},
                       {'lambda': 2.5}]
partial_info_ground = Partial_Info_Play_Ground(bandit_algorithms=[Linear_Gaussian_UCB(data_job, "Linear Gaussian UCB")],
                                              hyperparameters=[ucb_hyperparameters],
                                              plot_label = "High_Posterior",
                                              plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/",
                                              gap=MEDIUM_GAP)
partial_info_ground.plot_regret_as_function_of_hyperparameters_3(20)