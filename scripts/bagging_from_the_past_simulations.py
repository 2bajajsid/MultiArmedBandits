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
from bandit_algorithms.bagging_from_the_past import BaggingFromThePast_FI, BaggingFromThePast_FI_v0, BaggingFromThePast_PI, BaggingFromThePast_PI_v0, BaggingFromThePast_PI_v1
import numpy as np
import math

high_gap_stochastic_data_job = High_Gap_Stochastic()
low_gap_stochastic_data_job = Low_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()
stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch()

bp_fi = BaggingFromThePast_FI(stochastically_constrained_data_job)
bp_fi_v0 = BaggingFromThePast_FI_v0(stochastically_constrained_data_job)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [bp_fi, bp_fi_v0], 
                                          plot_label="Full Information Stochastic Adversarial (Exponential Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
full_info_ground.plot_results()

bp_fi = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job)
bp_fi_v0 = BaggingFromThePast_FI_v0(stochastically_constrained_fast_switch_data_job)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [bp_fi, bp_fi_v0], 
                                          plot_label="Full Information Stochastic Adversarial (Fast Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
full_info_ground.plot_results()

bp_fi = BaggingFromThePast_PI(stochastically_constrained_data_job)
bp_fi_v0 = BaggingFromThePast_PI_v0(stochastically_constrained_data_job)
bp_fi_v1 = BaggingFromThePast_PI_v1(stochastically_constrained_data_job)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [bp_fi, bp_fi_v0, bp_fi_v1], 
                                          plot_label="Partial Information Stochastic Adversarial (Exponential Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_ground.plot_results()

bp_fi = BaggingFromThePast_PI(stochastically_constrained_fast_switch_data_job)
bp_fi_v0 = BaggingFromThePast_PI_v0(stochastically_constrained_fast_switch_data_job)
bp_fi_v1 = BaggingFromThePast_PI_v1(stochastically_constrained_fast_switch_data_job)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                         [bp_fi, bp_fi_v0, bp_fi_v1], 
                                          plot_label="Partial Information Stochastic Adversarial (Fast Switch)",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_ground.plot_results()

bp_fi = BaggingFromThePast_PI(high_gap_stochastic_data_job)
bp_fi_v0 = BaggingFromThePast_PI_v0(high_gap_stochastic_data_job)
bp_fi_v1 = BaggingFromThePast_PI_v1(high_gap_stochastic_data_job)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                         [bp_fi, bp_fi_v0, bp_fi_v1], 
                                          plot_label="Partial Information High Gap",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_ground.plot_results()

bp_fi = BaggingFromThePast_PI(low_gap_stochastic_data_job)
bp_fi_v0 = BaggingFromThePast_PI_v0(low_gap_stochastic_data_job)
bp_fi_v1 = BaggingFromThePast_PI_v1(low_gap_stochastic_data_job)
partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                         [bp_fi, bp_fi_v0, bp_fi_v1], 
                                          plot_label="Partial Information Low Gap",
                                          plot_directory="results/bagging_from_the_past_simulations/")
partial_info_ground.plot_results()