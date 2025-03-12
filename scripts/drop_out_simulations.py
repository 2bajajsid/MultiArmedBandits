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
from bandit_algorithms.dropout import DropOut_FI, DropOut_PI
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

dropout_count_sqrt_t = lambda t : math.floor(math.sqrt(t))
dropout_count_log_t = lambda t : math.floor(math.log(t))
dropout_count_point_zero_five = lambda t : math.floor(0.05 * t)
dropout_count_point_one = lambda t : math.floor(0.1 * t)

dropout_1 = DropOut_FI(stochastically_constrained_data_job, get_dropout_count=dropout_count_sqrt_t, dropout_label="sqrt t")
dropout_2 = DropOut_FI(stochastically_constrained_data_job, get_dropout_count=dropout_count_log_t, dropout_label= "log t")
dropout_3 = DropOut_FI(stochastically_constrained_data_job, get_dropout_count=dropout_count_point_zero_five, dropout_label = "0.05 t")
dropout_4 = DropOut_FI(stochastically_constrained_data_job, get_dropout_count=dropout_count_point_one, dropout_label = "0.1 t")
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [dropout_1, dropout_2, dropout_3, dropout_4], 
                                          plot_label="Full Information Stochastic Adversarial (Exponential Switch)",
                                          plot_directory="results/dropout_simulations/")
full_info_ground.plot_results()

full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [dropout_1, dropout_2, dropout_3, dropout_4], 
                                          plot_label="Full Information Stochastic Adversarial (Fast Switch)",
                                          plot_directory="results/dropout_simulations/")
full_info_ground.plot_results()

dropout_1 = DropOut_PI(stochastically_constrained_data_job, get_dropout_count=dropout_count_sqrt_t, dropout_label="sqrt t")
dropout_2 = DropOut_PI(stochastically_constrained_data_job, get_dropout_count=dropout_count_log_t, dropout_label= "log t")
dropout_3 = DropOut_PI(stochastically_constrained_data_job, get_dropout_count=dropout_count_point_zero_five, dropout_label = "0.05 t")
dropout_4 = DropOut_PI(stochastically_constrained_data_job, get_dropout_count=dropout_count_point_one, dropout_label = "0.1 t")
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                               [dropout_1, dropout_2, dropout_3, dropout_4], 
                                               plot_label="Partial Information Stochastic Adversarial (Exponential Switch)",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                               [dropout_1, dropout_2, dropout_3, dropout_4], 
                                               plot_label="Partial Information Stochastic Adversarial (Fast Switch)",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()


partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [dropout_1, dropout_2, dropout_3, dropout_4], 
                                               plot_label="Partial Information High-Gap Stochastic",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                               [dropout_1, dropout_2, dropout_3, dropout_4], 
                                               plot_label="Partial Information Low-Gap Stochastic",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

