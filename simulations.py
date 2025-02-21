import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from high_gap_stochastic import High_Gap_Stochastic
from stochastically_constrained import Stochastically_Constrained
from partial_info_play_ground import Partial_Info_Play_Ground
from full_info_play_ground import Full_Info_Play_Ground
from ucb import UCB
from explore_then_commit import Explore_Then_Commit
from exp_3 import Exp3
from exp_3_plus_plus import Exp3_plus_plus
from tsallis_inf import Tsallis_Inf
from bagging_from_the_past import BaggingFromThePast_FI, BaggingFromThePast_PI, BaggingFromThePast_PI_v0, BaggingFromThePast_PI_v1
from dropout import DropOut_FI, DropOut_PI
from hedge import Hedge
from gaussian_thompson_sampling import Gaussian_Thompson_FI, Gaussian_Thompson_PI
import math
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()

hedge = Hedge(stochastically_constrained_data_job)
bagging_from_the_past_fi = BaggingFromThePast_FI(stochastically_constrained_data_job)
drop_out_fi = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.05)
gaussian_thompson_sampling = Gaussian_Thompson_FI(stochastically_constrained_data_job, sigma_sq=stochastically_constrained_data_job.get_K())
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [gaussian_thompson_sampling, hedge, bagging_from_the_past_fi, drop_out_fi], 
                                         plot_label="Stochastically Constrained Full Info Game")
full_info_ground.plot_results()

bagging_from_the_past = BaggingFromThePast_PI(high_gap_stochastic_data_job)
exp3 = Exp3(high_gap_stochastic_data_job)
exp3_plus_plus = Exp3_plus_plus(high_gap_stochastic_data_job)
tsallis_inf = Tsallis_Inf(high_gap_stochastic_data_job)
drop_out = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.05)
gaussian_thompson_sampling = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 
                                                  sigma_sq=stochastically_constrained_data_job.get_K())
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [gaussian_thompson_sampling, exp3, drop_out, 
                                                bagging_from_the_past, tsallis_inf, exp3_plus_plus], 
                                               plot_label="High Gap Stochastic Partial Info Game")
partial_info_ground.plot_results()

bagging_from_the_past = BaggingFromThePast_PI(stochastically_constrained_data_job)
exp3 = Exp3(stochastically_constrained_data_job)
exp3_plus_plus = Exp3_plus_plus(stochastically_constrained_data_job)
tsallis_inf = Tsallis_Inf(stochastically_constrained_data_job)
drop_out = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.05)
gaussian_thompson_sampling = Gaussian_Thompson_PI(stochastically_constrained_data_job, 
                                                  sigma_sq=stochastically_constrained_data_job.get_K())
partial_info_ground_2 = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [gaussian_thompson_sampling, exp3, drop_out, 
                                                 bagging_from_the_past, tsallis_inf, exp3_plus_plus],
                                                 plot_label = "Stochastically Constrained Partial Info Game")
partial_info_ground_2.plot_results()