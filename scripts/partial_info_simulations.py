import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from data_generating_mechanism.high_gap_stochastic import High_Gap_Stochastic
from data_generating_mechanism.stochastically_constrained import Stochastically_Constrained
from data_generating_mechanism.low_gap_stochastic import Low_Gap_Stochastic
from data_generating_mechanism.stochastically_constrained_fast_switch import Stochastically_Constrained_Fast_Switch
from play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from bandit_algorithms.bagging_from_the_past import BaggingFromThePast_PI
from bandit_algorithms.gaussian_thompson_sampling import Gaussian_Thompson_PI
from bandit_algorithms.follow_the_leader import Follow_The_Leader_PI
from bandit_algorithms.dropout import DropOut_PI
import math
import numpy as np

NUM_ARMS = 10
TIME_HORIZON = 1000
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

bag_size_log_t = lambda t : math.floor(math.log(t))
dropout_count_log_t = lambda t : math.floor(math.log(t))

gt_K_sqrt_t = Gaussian_Thompson_PI(stochastically_constrained_data_job, sigma_sq=math.sqrt(NUM_ARMS), sigma_sq_label="sqrt(K)")
ftl_fi = Follow_The_Leader_PI(stochastically_constrained_data_job)
bp_1 = BaggingFromThePast_PI(stochastically_constrained_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = False)
bp_2 = BaggingFromThePast_PI(stochastically_constrained_data_job, get_bag_size = bag_size_log_t, bag_label = "log(t)", add_bag = True)
dropout = DropOut_PI(stochastically_constrained_data_job, get_dropout_count=dropout_count_log_t, dropout_label= "log t")

partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [gt_K_sqrt_t, ftl_fi, bp_1, bp_2, dropout],
                                                 plot_label = "Stochastically Constrained (Exponential Switch)",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()


partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job,
                                                 [gt_K_sqrt_t, ftl_fi, bp_1, bp_2, dropout],
                                                 plot_label = "Stochastically Constrained (Fast Switch)",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()

partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job,
                                                 [gt_K_sqrt_t, ftl_fi, bp_1, bp_2, dropout],
                                                 plot_label = "Stochastic High Gap", 
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()


partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                                 [gt_K_sqrt_t, ftl_fi, bp_1, bp_2, dropout],
                                                 plot_label = "Stochastic Low Gap",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()