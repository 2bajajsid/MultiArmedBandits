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

high_gap_stochastic_data_job = High_Gap_Stochastic()
low_gap_stochastic_data_job = Low_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()
stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch()

gt_K = Gaussian_Thompson_FI(stochastically_constrained_data_job, stochastically_constrained_data_job.get_K())
gt_K_log = Gaussian_Thompson_FI(stochastically_constrained_data_job, math.log(stochastically_constrained_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, 0.01)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [gt_K, gt_K_log, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Full Information Stochastic Adversarial (Exponential Switch)",
                                          plot_directory="results/gaussian_ts_simulations/")
full_info_ground.plot_results()

gt_K = Gaussian_Thompson_FI(stochastically_constrained_fast_switch_data_job, stochastically_constrained_data_job.get_K())
gt_K_log = Gaussian_Thompson_FI(stochastically_constrained_fast_switch_data_job, math.log(stochastically_constrained_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_FI(stochastically_constrained_fast_switch_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_FI(stochastically_constrained_fast_switch_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_FI(stochastically_constrained_fast_switch_data_job, 0.01)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [gt_K, gt_K_log, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Full Information Stochastic Adversarial (Fast Switch)",
                                          plot_directory="results/gaussian_ts_simulations/")
full_info_ground.plot_results()

gt_K = Gaussian_Thompson_PI(high_gap_stochastic_data_job, high_gap_stochastic_data_job.get_K())
gt_K_log = Gaussian_Thompson_PI(high_gap_stochastic_data_job, math.log(high_gap_stochastic_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 0.01)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                         [gt_K, gt_K_log, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Partial Information High Gap",
                                          plot_directory="results/gaussian_ts_simulations/")
partial_info_ground.plot_results()

gt_K = Gaussian_Thompson_PI(low_gap_stochastic_data_job, low_gap_stochastic_data_job.get_K())
gt_K_log = Gaussian_Thompson_PI(low_gap_stochastic_data_job, math.log(low_gap_stochastic_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_PI(low_gap_stochastic_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_PI(low_gap_stochastic_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_PI(low_gap_stochastic_data_job, 0.01)
partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                         [gt_K, gt_K_log, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Partial Information Low Gap",
                                          plot_directory="results/gaussian_ts_simulations/")
partial_info_ground.plot_results()

gt_K = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 
                            stochastically_constrained_fast_switch_data_job.get_K())
gt_K_log = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 
                                math.log(stochastically_constrained_fast_switch_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 0.01)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [gt_K, gt_K_log, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="Partial Information Stochastically Constrained (Fast Switch)",
                                          plot_directory="results/gaussian_ts_simulations/")
partial_info_ground.plot_results()

