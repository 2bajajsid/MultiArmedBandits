import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from data_generating_mechanism.linear_gaussian_mechanism import Linear_Gaussian_Stochastic
from data_generating_mechanism.linear_posterior_sampling_mechanism import  Linear_Posterior_Sampling_Stochastic
from bandit_algorithms.linear_gaussian_ucb import Linear_Gaussian_UCB
from bandit_algorithms.linear_posterior_sampling import Linear_Posterior_Sampling
import numpy as np
import math

cov_matrix = np.random.rand(10, 10)
cov_matrix = cov_matrix.T @ cov_matrix + 0.001 * np.identity(10)
mean = np.random.uniform(-1, -5, size = 10)

linear_ps_data_job_1 = Linear_Posterior_Sampling_Stochastic()
linear_ps_data_job_2 = Linear_Posterior_Sampling_Stochastic(misspecify = True)

linear_thompson_sampling_1 = Linear_Posterior_Sampling(linear_ps_data_job_1)
linear_thompson_sampling_2 = Linear_Posterior_Sampling(linear_ps_data_job_2)

partial_info_ground = Partial_Info_Play_Ground(linear_ps_data_job_1, 
                                                 [linear_thompson_sampling_1, 
                                                  linear_thompson_sampling_2],
                                                 plot_label = "Linear-TS",
                                                 plot_directory = "/Users/sidbajaj/MultiArmedBandits/results/linear_ps_simulations/",
                                                 compute_importance_weighted_rewards = True)
partial_info_ground.plot_results()