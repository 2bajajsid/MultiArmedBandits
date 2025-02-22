import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from high_gap_stochastic import High_Gap_Stochastic
from stochastically_constrained import Stochastically_Constrained
from src.play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from full_info_play_ground import Full_Info_Play_Ground
from ucb import UCB
from explore_then_commit import Explore_Then_Commit
from exp_3 import Exp3
from exp_3_plus_plus import Exp3_plus_plus
from tsallis_inf import Tsallis_Inf
from src.bandit_algorithms.bagging_from_the_past import BaggingFromThePast_FI, BaggingFromThePast_PI, BaggingFromThePast_PI_v0, BaggingFromThePast_PI_v1
from dropout import DropOut_FI, DropOut_PI
from stochastically_constrained_fast_switch import Stochastically_Constrained_Fast_Switch
from hedge import Hedge
from gaussian_thompson_sampling import Gaussian_Thompson_FI, Gaussian_Thompson_PI
from low_gap_stochastic import Low_Gap_Stochastic
import math
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()
stochastically_constrained_fast_switch_data_job = Stochastically_Constrained_Fast_Switch()
low_gap_stochastic_data_job = Low_Gap_Stochastic()

bagging_from_the_past = BaggingFromThePast_PI(low_gap_stochastic_data_job)
exp3 = Exp3(low_gap_stochastic_data_job)
exp3_plus_plus = Exp3_plus_plus(low_gap_stochastic_data_job)
tsallis_inf = Tsallis_Inf(low_gap_stochastic_data_job)
drop_out = DropOut_PI(low_gap_stochastic_data_job, dropout_prob=0.01)
gaussian_thompson_sampling = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 
                                                  sigma_sq=1)
partial_info_ground = Partial_Info_Play_Ground(low_gap_stochastic_data_job, 
                                                 [gaussian_thompson_sampling, exp3, drop_out, 
                                                 bagging_from_the_past, tsallis_inf, exp3_plus_plus],
                                                 plot_label = "Stochastic Low Gap Partial Info Game")
partial_info_ground.plot_results()

bagging_from_the_past = BaggingFromThePast_PI(high_gap_stochastic_data_job)
exp3 = Exp3(high_gap_stochastic_data_job)
exp3_plus_plus = Exp3_plus_plus(high_gap_stochastic_data_job)
tsallis_inf = Tsallis_Inf(high_gap_stochastic_data_job)
drop_out = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.05)
gaussian_thompson_sampling = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 
                                                  sigma_sq=1)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [gaussian_thompson_sampling, exp3, drop_out, 
                                                bagging_from_the_past, tsallis_inf, exp3_plus_plus], 
                                               plot_label="High Gap Stochastic Partial Info Game")
partial_info_ground.plot_results()

bagging_from_the_past = BaggingFromThePast_PI(stochastically_constrained_data_job)
exp3 = Exp3(stochastically_constrained_data_job)
exp3_plus_plus = Exp3_plus_plus(stochastically_constrained_data_job)
tsallis_inf = Tsallis_Inf(stochastically_constrained_data_job)
drop_out = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.01)
gaussian_thompson_sampling = Gaussian_Thompson_PI(stochastically_constrained_data_job, 
                                                  sigma_sq=1)
partial_info_ground_2 = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [gaussian_thompson_sampling, exp3, drop_out, 
                                                 bagging_from_the_past, tsallis_inf, exp3_plus_plus],
                                                 plot_label = "Stochastically Constrained Partial Info Game (Exponential Time Switch)")
partial_info_ground_2.plot_results()

bagging_from_the_past = BaggingFromThePast_PI(stochastically_constrained_fast_switch_data_job)
exp3 = Exp3(stochastically_constrained_fast_switch_data_job)
exp3_plus_plus = Exp3_plus_plus(stochastically_constrained_fast_switch_data_job)
tsallis_inf = Tsallis_Inf(stochastically_constrained_fast_switch_data_job)
drop_out = DropOut_PI(stochastically_constrained_fast_switch_data_job, dropout_prob=0.01)
gaussian_thompson_sampling = Gaussian_Thompson_PI(stochastically_constrained_fast_switch_data_job, 
                                                  sigma_sq=1)
partial_info_ground_2 = Partial_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                                 [gaussian_thompson_sampling, exp3, drop_out, 
                                                 bagging_from_the_past, tsallis_inf, exp3_plus_plus],
                                                 plot_label = "Stochastically Constrained Partial Info Game (Faster Switch)")
partial_info_ground_2.plot_results()