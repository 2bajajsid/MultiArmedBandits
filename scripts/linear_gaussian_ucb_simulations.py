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
np.random.seed(0)

linear_gaussian_stochastic_data_job_1 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.5, post = 10)
linear_gaussian_stochastic_data_job_2 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.1, post = 10)
linear_gaussian_stochastic_data_job_3 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.2, post = 10)
linear_gaussian_stochastic_data_job_4 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.3, post = 10)
linear_gaussian_stochastic_data_job_5 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.4, post = 10)
linear_gaussian_stochastic_data_job_6 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.75, post = 10)

linear_gaussian_ucb_1 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_1)
linear_gaussian_ucb_2 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_2)
linear_gaussian_ucb_3 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_3)
linear_gaussian_ucb_4 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_4)
linear_gaussian_ucb_5 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_5)
linear_gaussian_ucb_6 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_6)

partial_info_ground = Partial_Info_Play_Ground(linear_gaussian_stochastic_data_job_1, 
                                                 [linear_gaussian_ucb_2],
                                                 plot_label = "Linear-Gaussian-UCB-10",
                                                 plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/linear_gaussian_ucb_simulations/",
                                                 compute_importance_weighted_rewards = True)
partial_info_ground.plot_results()


linear_gaussian_stochastic_data_job_1 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.15, post = 2)
linear_gaussian_stochastic_data_job_2 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.1, post = 2)
linear_gaussian_stochastic_data_job_3 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.25, post = 2)
linear_gaussian_stochastic_data_job_4 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.5, post = 2)
linear_gaussian_stochastic_data_job_5 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.75, post = 2)
linear_gaussian_stochastic_data_job_6 = Linear_Gaussian_Stochastic(delta = 0.001, lambda_reg = 0.9, post = 2)

linear_gaussian_ucb_1 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_1)
linear_gaussian_ucb_2 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_2)
linear_gaussian_ucb_3 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_3)
linear_gaussian_ucb_4 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_4)
linear_gaussian_ucb_5 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_5)
linear_gaussian_ucb_6 = Linear_Gaussian_UCB(linear_gaussian_stochastic_data_job_6)

partial_info_ground = Partial_Info_Play_Ground(linear_gaussian_stochastic_data_job_1, 
                                                 [linear_gaussian_ucb_4],
                                                 plot_label = "Linear-Gaussian-UCB-2",
                                                 plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/linear_gaussian_ucb_simulations/",
                                                 compute_importance_weighted_rewards = True)
partial_info_ground.plot_results()