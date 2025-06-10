import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from data_generating_mechanism.linear_gaussian_mechanism import Linear_Gaussian_Stochastic
from bandit_algorithms.linear_gaussian_ucb import Linear_Gaussian_UCB
import numpy as np
import math

linear_gaussian_stochastic_data_job = Linear_Gaussian_Stochastic()
linear_gaussian_ucb = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job)

partial_info_ground = Partial_Info_Play_Ground(linear_gaussian_stochastic_data_job, 
                                                 [linear_gaussian_ucb],
                                                 plot_label = "Linear-Gaussian-UCB",
                                                 plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/linear_gaussian_ucb_simulations/",
                                                 compute_importance_weighted_rewards = True)
partial_info_ground.plot_results()