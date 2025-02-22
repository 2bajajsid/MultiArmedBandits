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
from bandit_algorithms.dropout_log import DropOutLog_FI, DropOutLog_PI
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic()
low_gap_stochastic_data_job = Low_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()
stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch()

drop_out_point_zero_one = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.05)
drop_out_log_one = DropOutLog_FI(stochastically_constrained_data_job, p=1)
drop_out_log_point_two = DropOutLog_FI(stochastically_constrained_data_job, p=0.2)
drop_out_log_point_zero_five = DropOutLog_FI(stochastically_constrained_data_job, p=0.05)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [drop_out_point_zero_one, drop_out_point_zero_five, 
                                          drop_out_log_one, drop_out_log_point_two, drop_out_log_point_zero_five], 
                                          plot_label="Full Information Stochastic Adversarial (Exponential Switch)",
                                          plot_directory="results/dropout_simulations/")
full_info_ground.plot_results()

drop_out_point_zero_one = DropOut_FI(stochastically_constrained_fast_switch_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_FI(stochastically_constrained_fast_switch_data_job, dropout_prob=0.05)
drop_out_point_one = DropOut_FI(stochastically_constrained_fast_switch_data_job, dropout_prob=0.1)
drop_out_log_one = DropOutLog_FI(stochastically_constrained_fast_switch_data_job, p=1)
drop_out_log_point_two = DropOutLog_FI(stochastically_constrained_fast_switch_data_job, p=0.2)
drop_out_log_point_zero_five = DropOutLog_FI(stochastically_constrained_fast_switch_data_job, p=0.05)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [drop_out_point_zero_one, drop_out_point_zero_five,
                                          drop_out_log_one, drop_out_log_point_two, drop_out_log_point_zero_five], 
                                          plot_label="Full Information Stochastic Adversarial (Fast Switch)",
                                          plot_directory="results/dropout_simulations/")
full_info_ground.plot_results()

drop_out_point_zero_one = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.05)
drop_out_log_one = DropOutLog_PI(stochastically_constrained_data_job, p=1)
drop_out_log_point_two = DropOutLog_PI(stochastically_constrained_data_job, p=0.2)
drop_out_log_point_zero_five = DropOutLog_PI(stochastically_constrained_data_job, p=0.05)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                               [drop_out_point_zero_one, drop_out_point_zero_five,  
                                                drop_out_log_one, drop_out_log_point_two, drop_out_log_point_zero_five], 
                                               plot_label="Partial Information Stochastic Adversarial (Exponential Switch)",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

drop_out_point_zero_one = DropOut_PI(stochastically_constrained_fast_switch_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_PI(stochastically_constrained_fast_switch_data_job, dropout_prob=0.05)
drop_out_log_one = DropOutLog_PI(stochastically_constrained_fast_switch_data_job, p=1)
drop_out_log_point_two = DropOutLog_PI(stochastically_constrained_fast_switch_data_job, p=0.2)
drop_out_log_point_zero_five = DropOutLog_PI(stochastically_constrained_fast_switch_data_job, p=0.05)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                               [drop_out_point_zero_one, drop_out_point_zero_five,  
                                                drop_out_log_one, drop_out_log_point_two, drop_out_log_point_zero_five], 
                                               plot_label="Partial Information Stochastic Adversarial (Fast Switch)",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

drop_out_point_zero_one = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.05)
drop_out_log_one = DropOutLog_PI(high_gap_stochastic_data_job, p=1)
drop_out_log_point_two = DropOutLog_PI(high_gap_stochastic_data_job, p=0.2)
drop_out_log_point_zero_five = DropOutLog_PI(high_gap_stochastic_data_job, p=0.05)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [drop_out_point_zero_one, drop_out_point_zero_five,  
                                                drop_out_log_one, drop_out_log_point_two, drop_out_log_point_zero_five], 
                                               plot_label="Partial Information High-Gap Stochastic",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

drop_out_point_zero_one = DropOut_PI(low_gap_stochastic_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_PI(low_gap_stochastic_data_job, dropout_prob=0.05)
drop_out_log_one = DropOutLog_PI(low_gap_stochastic_data_job, p=1)
drop_out_log_point_two = DropOutLog_PI(low_gap_stochastic_data_job, p=0.2)
drop_out_log_point_zero_five = DropOutLog_PI(low_gap_stochastic_data_job, p=0.05)
partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                               [drop_out_point_zero_one, drop_out_point_zero_five,  
                                                drop_out_log_one, drop_out_log_point_two, drop_out_log_point_zero_five], 
                                               plot_label="Partial Information Low-Gap Stochastic",
                                               plot_directory="results/dropout_simulations/")
partial_info_ground.plot_results()

