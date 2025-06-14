import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from data_generating_mechanism.linear_gaussian_mechanism import Linear_Gaussian_Stochastic
from data_generating_mechanism.linear_posterior_sampling_mechanism import Linear_Posterior_Sampling_Stochastic
from data_generating_mechanism.ucb_mixture_mechanism import UCB_Mixture_Mechanism
from data_generating_mechanism.posterior_sampling_mechanism import Posterior_Sampling_Stochastic
from bandit_algorithms.linear_gaussian_ucb import Linear_Gaussian_UCB
from bandit_algorithms.linear_posterior_sampling import Linear_Posterior_Sampling
from game.partial_information_game import Partial_Information_Game
from scipy.optimize import Bounds
import numpy as np
import math
np.random.seed(0)

'''
linear_ps_data_job_1 = Linear_Posterior_Sampling_Stochastic()
linear_thompson_sampling = Linear_Posterior_Sampling(linear_ps_data_job_1)

ucb_algo = Linear_Gaussian_UCB(UCB_Mixture_Mechanism())
ucb_hyperparameters = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
partial_info_ground = Partial_Info_Play_Ground(bandit_algorithms=[ucb_algo],
                                              hyperparameters = [ucb_hyperparameters],
                                              plot_label = "Mixture-Sims",
                                              plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/linear_gaussian_ucb_simulations/")
partial_info_ground.plot_results()
partial_info_ground.games[0].find_minimum([0.3], Bounds([0], [1]))
'''

linear_ps_data_job_1 = Linear_Posterior_Sampling_Stochastic()
linear_thompson_sampling = Linear_Posterior_Sampling(linear_ps_data_job_1)

linear_gaussian_stochastic_data_job_1 = Linear_Gaussian_Stochastic()
linear_gaussian_ucb_1 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_1)

gaussian_ucb_values = [{'lambda': 1.0, 'delta': 0.001},
                       {'lambda': 0.1, 'delta': 0.001}, 
                       {'lambda': 0.1885, 'delta': 0.0083},
                       {'lambda': 0.676, 'delta': 0.001}]

ts_hyperparameter = [[np.zeros(10), 10 * np.identity(10)]]

'''
ucb_algo = Linear_Gaussian_UCB(UCB_Mixture_Mechanism())
ucb_hyperparameters = [{'delta': 0.00001},
                       {'delta': 0.04335},
                       {'delta': 0.1},
                       {'delta': 0.25},
                       {'delta': 0.35}]

ps_algo = Linear_Posterior_Sampling(Posterior_Sampling_Stochastic())
ps_algo_hyperparameters = [{'gamma': 0},
                           {'gamma': 0.1},
                           {'gamma': 0.2},
                           {'gamma': 0.3}, 
                           {'gamma': 0.4}, 
                           {'gamma': 0.5}, 
                           {'gamma': 0.6},
                           {'gamma': 0.7},
                           {'gamma': 0.8}, 
                           {'gamma': 0.9},
                           {'gamma': 1.0}]
'''

partial_info_ground = Partial_Info_Play_Ground(bandit_algorithms=[linear_thompson_sampling, 
                                                                  linear_gaussian_ucb_1],
                                              hyperparameters=[ts_hyperparameter, gaussian_ucb_values],
                                              plot_label = "Linear-Gaussian-UCB",
                                              plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/linear_gaussian_ucb_simulations/")
partial_info_ground.plot_results()
#partial_info_ground.games[0].find_minimum([0.3], Bounds([0], [1]))

'''
grid_values = [[0.01,0.001], 
               [0.1,0.001], 
               [0.3,0.001], 
               [0.5,0.001], 
               [0.7,0.001], 
               [1.1,0.001], 
               [1.5,0.001], 
               [4,0.001],
               [10,0.001]]
linear_gaussian_stochastic_data_job_1 = Linear_Gaussian_Stochastic()
linear_gaussian_ucb_1 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_1)

pi = Partial_Information_Game(bandit_algorithm= linear_gaussian_ucb_1)
pi.grid_search(grid_values=grid_values)
'''