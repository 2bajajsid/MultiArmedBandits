import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from high_gap_stochastic import High_Gap_Stochastic
from stochastically_constrained import Stochastically_Constrained
from partial_info_play_ground import Partial_Info_Play_Ground
from ucb import UCB
from explore_then_commit import Explore_Then_Commit
from exp_3 import Exp3
from exp_3_plus_plus import Exp3_plus_plus
from tsallis_inf import Tsallis_Inf
import math

high_gap_stochastic_data_job = High_Gap_Stochastic()
ucb_algorithm = UCB(high_gap_stochastic_data_job)
explore_then_commit_algo = Explore_Then_Commit(high_gap_stochastic_data_job)
exp3 = Exp3(high_gap_stochastic_data_job)
exp3_plus_plus = Exp3_plus_plus(high_gap_stochastic_data_job)
tsallis_inf = Tsallis_Inf(high_gap_stochastic_data_job)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [tsallis_inf, ucb_algorithm, explore_then_commit_algo, exp3, exp3_plus_plus], 
                                               plot_label="High Gap Stochastic Partial Info Game")
partial_info_ground.plot_results()

stochastically_constrained_data_job = Stochastically_Constrained()
ucb_algorithm = UCB(stochastically_constrained_data_job)
explore_then_commit_algo = Explore_Then_Commit(stochastically_constrained_data_job)
exp3 = Exp3(stochastically_constrained_data_job)
exp3_plus_plus = Exp3_plus_plus(stochastically_constrained_data_job)
tsallis_inf = Tsallis_Inf(stochastically_constrained_data_job)
partial_info_ground_2 = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 [tsallis_inf, ucb_algorithm, explore_then_commit_algo, exp3, exp3_plus_plus],
                                                 plot_label = "Stochastically Constrained Partial Info Game")
partial_info_ground_2.plot_results()