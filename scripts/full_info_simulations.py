
import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from data_generating_mechanism.high_gap_stochastic import High_Gap_Stochastic
from data_generating_mechanism.low_gap_stochastic import Low_Gap_Stochastic
from data_generating_mechanism.stochastically_constrained import Stochastically_Constrained
from data_generating_mechanism.stochastically_constrained_fast_switch import Stochastically_Constrained_Fast_Switch
from play_ground.full_info_play_ground import Full_Info_Play_Ground
from bandit_algorithms.bagging_from_the_past import BaggingFromThePast_FI
from bandit_algorithms.gaussian_thompson_sampling import Gaussian_Thompson_FI
from bandit_algorithms.follow_the_leader import Follow_The_Leader_FI
from bandit_algorithms.dropout import DropOut_FI
import math
import numpy as np

NUM_ARMS = 50
TIME_HORIZON = 5000
INIT_EXPLORATION = 0
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

bag_size_log_t = lambda t : math.floor(math.log(t))
dropout_count_sqrt_t = lambda t : math.floor(math.sqrt(t))
dropout_count_log_t = lambda t : math.floor(math.log(t))

gt_K_point_zero_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, sigma_sq=0.01, sigma_sq_label="0.01")
ftl_fi = Follow_The_Leader_FI(stochastically_constrained_data_job)
bp_1 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = False)
bp_2 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = True)
dropout_1 = DropOut_FI(stochastically_constrained_data_job, get_dropout_count=dropout_count_sqrt_t, dropout_label="sqrt t")
dropout_2 = DropOut_FI(stochastically_constrained_data_job, get_dropout_count=dropout_count_log_t, dropout_label= "log t")

full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [gt_K_point_zero_1, ftl_fi, bp_1, bp_2, dropout_1, dropout_2],
                                                 plot_label = "Stochastically Constrained (Exponential Switch)",
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()


full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                                 [gt_K_point_zero_1, ftl_fi, bp_1, bp_2, dropout_1, dropout_2],
                                                 plot_label = "Stochastically Constrained (Fast Switch)",
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()

full_info_ground = Full_Info_Play_Ground(high_gap_stochastic_data_job, 
                                                 [gt_K_point_zero_1, ftl_fi, bp_1, bp_2, dropout_1, dropout_2],
                                                 plot_label = "Stochastic High Gap", 
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()


full_info_ground = Full_Info_Play_Ground(low_gap_stochastic_data_job, 
                                                 [gt_K_point_zero_1, ftl_fi, bp_1, bp_2, dropout_1, dropout_2],
                                                 plot_label = "Stochastic Low Gap",
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()