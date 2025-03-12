import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from play_ground.full_info_play_ground import Full_Info_Play_Ground
from data_generating_mechanism.high_gap_stochastic import High_Gap_Stochastic
from data_generating_mechanism.stochastically_constrained import Stochastically_Constrained
from data_generating_mechanism.stochastically_constrained_fast_switch import Stochastically_Constrained_Fast_Switch
from data_generating_mechanism.low_gap_stochastic import Low_Gap_Stochastic
from bandit_algorithms.gaussian_thompson_sampling import Gaussian_Thompson_FI, Gaussian_Thompson_PI
import numpy as np
import math

NUM_ARMS = 25
TIME_HORIZON = 2500
INIT_EXPLORATION = 5
high_gap_stochastic_data_job = High_Gap_Stochastic(num_arms = NUM_ARMS, 
                                                   time_horizon = TIME_HORIZON, 
                                                   init_exploration = INIT_EXPLORATION)

stochastically_constrained_data_job = Stochastically_Constrained(num_arms = NUM_ARMS, 
                                                                 time_horizon = TIME_HORIZON, 
                                                                 init_exploration = INIT_EXPLORATION)

stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch(num_arms = NUM_ARMS, 
                                                                                         time_horizon = TIME_HORIZON, 
                                                                                         init_exploration = INIT_EXPLORATION)

low_gap_stochastic_data_job = Low_Gap_Stochastic(num_arms = NUM_ARMS, 
                                                 time_horizon = TIME_HORIZON, 
                                                 init_exploration = INIT_EXPLORATION)

gt_sqrt_K = Gaussian_Thompson_FI(stochastically_constrained_data_job, math.sqrt(NUM_ARMS), sigma_sq_label="sqrt(K)")
gt_log_K = Gaussian_Thompson_FI(stochastically_constrained_data_job, math.log(NUM_ARMS), sigma_sq_label="log(K)")
gt_K_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, sigma_sq=1, sigma_sq_label = "1")
gt_K_point_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, sigma_sq=0.1, sigma_sq_label="0.1")
gt_K_point_zero_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, sigma_sq=0.01, sigma_sq_label="0.01")
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [gt_sqrt_K, gt_log_K, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Full Information Stochastic Adversarial (Exponential Switch)",
                                          plot_directory="results/gaussian_ts_simulations/")
full_info_ground.plot_results()

full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [gt_sqrt_K, gt_log_K, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Full Information Stochastic Adversarial (Fast Switch)",
                                          plot_directory="results/gaussian_ts_simulations/")
full_info_ground.plot_results()

gt_sqrt_K = Gaussian_Thompson_PI(stochastically_constrained_data_job, math.sqrt(NUM_ARMS), sigma_sq_label="sqrt(K)")
gt_log_K = Gaussian_Thompson_PI(stochastically_constrained_data_job, math.log(NUM_ARMS), sigma_sq_label="log(K)")
gt_K_1 = Gaussian_Thompson_PI(stochastically_constrained_data_job, sigma_sq=1, sigma_sq_label = "1")
gt_K_point_1 = Gaussian_Thompson_PI(stochastically_constrained_data_job, sigma_sq=0.1, sigma_sq_label="0.1")
gt_K_point_zero_1 = Gaussian_Thompson_PI(stochastically_constrained_data_job, sigma_sq=0.01, sigma_sq_label="0.01")

partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                         [gt_sqrt_K, gt_log_K, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Partial Information High Gap",
                                          plot_directory="results/gaussian_ts_simulations/")
partial_info_ground.plot_results()

partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                         [gt_sqrt_K, gt_log_K, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Partial Information Low Gap",
                                          plot_directory="results/gaussian_ts_simulations/")
partial_info_ground.plot_results()

partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [gt_sqrt_K, gt_log_K, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Partial Information Stochastically Constrained (Fast Switch)",
                                          plot_directory="results/gaussian_ts_simulations/")
partial_info_ground.plot_results()

