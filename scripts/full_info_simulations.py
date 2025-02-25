
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
from bandit_algorithms.hedge import Hedge
from bandit_algorithms.gaussian_thompson_sampling import Gaussian_Thompson_FI
from bandit_algorithms.dropout_log import DropOutLog_FI
import math
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic(num_arms = 100, time_horizon = 5000, init_exploration = 0)
stochastically_constrained_data_job = Stochastically_Constrained(num_arms = 100, time_horizon = 5000, init_exploration = 0)
stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch(num_arms = 100, time_horizon = 5000, init_exploration = 0)
low_gap_stochastic_data_job = Low_Gap_Stochastic(num_arms = 100, time_horizon = 5000, init_exploration = 0)

hedge = Hedge(stochastically_constrained_data_job)
bagging_from_the_past = BaggingFromThePast_FI(stochastically_constrained_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_FI(stochastically_constrained_data_job, 
                                                  sigma_sq=0.01)
drop_out_log = DropOutLog_FI(stochastically_constrained_data_job, p=1)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log, hedge],
                                                 plot_label = "Stochastically Constrained (Exponential Switch)",
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()

hedge = Hedge(stochastically_constrained_fast_switch_data_job)
bagging_from_the_past = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_FI(stochastically_constrained_fast_switch_data_job, 
                                                  sigma_sq=0.01)
drop_out_log = DropOutLog_FI(stochastically_constrained_fast_switch_data_job, p=1)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log, hedge],
                                                 plot_label = "Stochastically Constrained (Fast Switch)",
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()

hedge = Hedge(high_gap_stochastic_data_job)
bagging_from_the_past = BaggingFromThePast_FI(high_gap_stochastic_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_FI(high_gap_stochastic_data_job, 
                                                  sigma_sq=0.01)
drop_out_log = DropOutLog_FI(high_gap_stochastic_data_job, p=1)
full_info_ground = Full_Info_Play_Ground(high_gap_stochastic_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log, hedge],
                                                 plot_label = "Stochastic High Gap", 
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()

hedge = Hedge(low_gap_stochastic_data_job)
bagging_from_the_past = BaggingFromThePast_FI(low_gap_stochastic_data_job)
gaussian_thompson_sampling = Gaussian_Thompson_FI(low_gap_stochastic_data_job, 
                                                  sigma_sq=0.01)
drop_out_log = DropOutLog_FI(low_gap_stochastic_data_job, p=1)
full_info_ground = Full_Info_Play_Ground(low_gap_stochastic_data_job, 
                                                 [bagging_from_the_past, gaussian_thompson_sampling, drop_out_log, hedge],
                                                 plot_label = "Stochastic Low Gap",
                                                 plot_directory = "results/full_info_simulations/")
full_info_ground.plot_results()