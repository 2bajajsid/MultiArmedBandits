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
from bandit_algorithms.exp_3_plus_plus import Exp3_plus_plus
from bandit_algorithms.tsallis_inf import Tsallis_Inf
from bandit_algorithms.gaussian_thompson_sampling import Gaussian_Thompson_PI
from bandit_algorithms.dropout_log import DropOutLog_PI
import math
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic(num_arms = 50, time_horizon = 3000, init_exploration = 5)
stochastically_constrained_data_job = Stochastically_Constrained(num_arms = 50, time_horizon = 3000, init_exploration = 5)
stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch(num_arms = 50, time_horizon = 3000, init_exploration = 5)
low_gap_stochastic_data_job = Low_Gap_Stochastic(num_arms = 50, time_horizon = 3000, init_exploration = 5)

tsallis_inf = Tsallis_Inf(stochastically_constrained_data_job)
exp3_plus_plus = Exp3_plus_plus(stochastically_constrained_data_job)
bagging_from_the_past = BaggingFromThePast_PI(stochastically_constrained_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_PI(stochastically_constrained_data_job, 
                                                  sigma_sq=math.log(100))
drop_out_log = DropOutLog_PI(stochastically_constrained_data_job, p=1)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log,
                                                  exp3_plus_plus, tsallis_inf],
                                                 plot_label = "Stochastically Constrained (Exponential Switch)",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()

tsallis_inf = Tsallis_Inf(stochastically_constrained_fast_switch_data_job)
exp3_plus_plus = Exp3_plus_plus(stochastically_constrained_fast_switch_data_job)
bagging_from_the_past = BaggingFromThePast_PI(stochastically_constrained_fast_switch_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 
                                                  sigma_sq=math.log(100))
drop_out_log = DropOutLog_PI(stochastically_constrained_fast_switch_data_job, p=1)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log,
                                                  exp3_plus_plus, tsallis_inf],
                                                 plot_label = "Stochastically Constrained (Fast Switch)",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()

tsallis_inf = Tsallis_Inf(high_gap_stochastic_data_job)
exp3_plus_plus = Exp3_plus_plus(high_gap_stochastic_data_job)
bagging_from_the_past = BaggingFromThePast_PI(high_gap_stochastic_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 
                                                  sigma_sq=math.log(100))
drop_out_log = DropOutLog_PI(high_gap_stochastic_data_job, p=1)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log,
                                                  exp3_plus_plus, tsallis_inf],
                                                 plot_label = "Stochastic High Gap",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()

tsallis_inf = Tsallis_Inf(low_gap_stochastic_data_job)
exp3_plus_plus = Exp3_plus_plus(low_gap_stochastic_data_job)
bagging_from_the_past = BaggingFromThePast_PI(low_gap_stochastic_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_PI(low_gap_stochastic_data_job, 
                                                  sigma_sq=math.log(100))
drop_out_log = DropOutLog_PI(low_gap_stochastic_data_job, p=1)
partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log,
                                                  exp3_plus_plus, tsallis_inf],
                                                 plot_label = "Stochastic Low Gap",
                                                 plot_directory = "results/partial_info_simulations/")
partial_info_ground.plot_results()