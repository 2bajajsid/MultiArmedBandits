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
from bandit_algorithms.bagging_from_the_past import BaggingFromThePast_PI, BaggingFromThePast_FI, BaggingFromThePast_Cache_FI, BaggingFromThePast_Cache_PI
import numpy as np
import math

NUM_ARMS = 25
TIME_HORIZON = 2500
INIT_EXPLORATION = 10
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

bag_size_sqrt_t = lambda t : math.floor(math.sqrt(t))
bag_size_log_t = lambda t : math.floor(math.log(t))
bag_size_100 = lambda t : 100

bp_1 = BaggingFromThePast_PI(high_gap_stochastic_data_job, get_bag_size = bag_size_100, bag_label = "100", add_bag = True)
bp_2 = BaggingFromThePast_Cache_PI(high_gap_stochastic_data_job, get_bag_size = bag_size_100, bag_label  = "100", add_bag = True)
bp_3 = BaggingFromThePast_PI(high_gap_stochastic_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = True)
bp_4 = BaggingFromThePast_PI(high_gap_stochastic_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = False)
bp_5 = BaggingFromThePast_Cache_PI(high_gap_stochastic_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = False)

partial_info_play_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Partial Info High Gap",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_play_ground.plot_results()


partial_info_play_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Partial Info Low Gap",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_play_ground.plot_results()

partial_info_play_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Partial Info Stochastically Constrained (Exponential Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_play_ground.plot_results()

partial_info_play_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Partial Info Stochastically Constrained (Fast Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_play_ground.plot_results()

full_info_ground = Full_Info_Play_Ground(high_gap_stochastic_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Full Info High Gap",
                                          plot_directory="results/bagging_from_the_past_simulations/")
full_info_ground.plot_results()


full_info_ground = Full_Info_Play_Ground(low_gap_stochastic_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Full Info Low Gap",
                                          plot_directory="results/bagging_from_the_past_simulations/")
full_info_ground.plot_results()

full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Full Info Stochastically Constrained (Exponential Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
full_info_ground.plot_results()

full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [bp_1, bp_2, bp_3, bp_4, bp_5], 
                                          plot_label="Full Info Stochastically Constrained (Fast Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
full_info_ground.plot_results()


