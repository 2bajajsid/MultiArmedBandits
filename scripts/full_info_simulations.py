
import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from data_generating_mechanism.high_gap_stochastic import High_Gap_Stochastic
from data_generating_mechanism.low_gap_stochastic import Low_Gap_Stochastic
from data_generating_mechanism.stochastically_constrained import Stochastically_Constrained
from data_generating_mechanism.stochastically_constrained_fast_switch import Stochastically_Constrained_Fast_Switch
from data_generating_mechanism.flip_flop_mechanism import Flip_Flop_Mechanism
from data_generating_mechanism.low_rank_projection import Low_Rank_Stochastic
from play_ground.full_info_play_ground import Full_Info_Play_Ground
from bandit_algorithms.bagging_from_the_past import BaggingFromThePast_FI
from bandit_algorithms.follow_the_leader import Follow_The_Leader_FI
from bandit_algorithms.hedge import Hedge
from bandit_algorithms.gaussian_ftpl import Gaussian_FTPL_FI
from bandit_algorithms.student_t_ftpl import Student_T_FTPL_FI
from bandit_algorithms.ucb import UCB
import math
import numpy as np
from threading import Thread
import multiprocessing as mp

NUM_ARMS = 15
MONTE_CARLO_RUNS = 100

def target_func(data_job, plot_label):
    gaussian_ftpl_scaled_down = Gaussian_FTPL_FI(data_job, scaled_down = True, num_arms = NUM_ARMS)
    gaussian_ftpl_identity_cov_scaled_down = Gaussian_FTPL_FI(data_job, identity_cov = True, scaled_down=True, num_arms = NUM_ARMS)
    gaussian_ftpl_scaled_down_zero_mean = Gaussian_FTPL_FI(data_job, zero_mean=True, scaled_down = True, num_arms = NUM_ARMS)
    gaussian_ftpl_identity_cov_scaled_down_zero_mean = Gaussian_FTPL_FI(data_job, zero_mean = True, identity_cov = True, scaled_down=True, num_arms = NUM_ARMS)

    algorithms = [
        gaussian_ftpl_scaled_down,
        gaussian_ftpl_identity_cov_scaled_down,
        gaussian_ftpl_scaled_down_zero_mean,
        gaussian_ftpl_identity_cov_scaled_down_zero_mean
    ]

    full_info_ground = Full_Info_Play_Ground(data_job, 
                                         algorithms, 
                                         plot_label = plot_label,
                                         plot_directory="rewards")
    full_info_ground.plot_regret()

data_jobs = [Low_Rank_Stochastic(num_arms = NUM_ARMS, low_rank_dimension=8, M = MONTE_CARLO_RUNS),
             High_Gap_Stochastic(num_arms = NUM_ARMS, M = MONTE_CARLO_RUNS)]

plot_labels = ["Low Rank Projection (dimension 10)",
               "High Gap Stochastic"]

if __name__ == '__main__':
    ctx = mp.get_context('spawn')
    
    threads = []

    for i in range(len(data_jobs)):
        threads.append(ctx.Process(target=target_func, 
                                    args = (data_jobs[i],
                                            plot_labels[i])))

    for t in threads:
        t.start()

    for t in threads:
        t.join()

'''
bp_1 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = True)
bp_2 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = False)
bp_3 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = True)
bp_4 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = False)
bp_5 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = True)
bp_6 = BaggingFromThePast_FI(stochastically_constrained_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = False)


student_t = Student_T_FTPL_FI(stochastically_constrained_data_job)
hedge = Hedge(stochastically_constrained_data_job)
ftl = Follow_The_Leader_FI(stochastically_constrained_data_job)

gaussian_ftpl = Gaussian_FTPL_FI(stochastically_constrained_data_job)
gaussian_ftpl_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_data_job, scaled_down = True)
gaussian_ftpl_identity_cov_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_data_job, zero_mean = True, scaled_down = True)

guassian_ftpl_identity_cov_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_data_job, identity_cov = True)
gaussian_ftpl_zero_mean = Gaussian_FTPL_FI(stochastically_constrained_data_job, zero_mean = True)
gaussian_ftpl_inflated_mean_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_data_job, inflated_mean = True, scaled_down = True)

ucb = UCB(stochastically_constrained_data_job)

algorithms = [gaussian_ftpl,
              gaussian_ftpl_scaled_down,
              bp_1,bp_5]


full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                                 algorithms,
                                                 plot_label = "Stochastically Constrained (Exponential Switch)",
                                                 plot_directory = "results/full_info_simulations/rewards/")
full_info_ground.plot_regret()


bp_1 = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = True)
bp_2 = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = False)
bp_3 = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = True)
bp_4 = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = False)
bp_5 = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = True)
bp_6 = BaggingFromThePast_FI(stochastically_constrained_fast_switch_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = False)

student_t = Student_T_FTPL_FI(stochastically_constrained_fast_switch_data_job)
hedge = Hedge(stochastically_constrained_fast_switch_data_job)
ftl = Follow_The_Leader_FI(stochastically_constrained_fast_switch_data_job)

gaussian_ftpl = Gaussian_FTPL_FI(stochastically_constrained_fast_switch_data_job)
gaussian_ftpl_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_fast_switch_data_job, scaled_down = True)
gaussian_ftpl_zero_mean = Gaussian_FTPL_FI(stochastically_constrained_fast_switch_data_job, zero_mean = True)
gaussian_ftpl_zero_mean_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_fast_switch_data_job, zero_mean = True, scaled_down = True)
gaussian_ftpl_identity_cov = Gaussian_FTPL_FI(stochastically_constrained_fast_switch_data_job, identity_cov = True)
gaussian_ftpl_inflated_mean_scaled_down = Gaussian_FTPL_FI(stochastically_constrained_fast_switch_data_job, inflated_mean = True, scaled_down = True)
ucb = UCB(stochastically_constrained_fast_switch_data_job)

algorithms = [gaussian_ftpl,
              gaussian_ftpl_scaled_down,
              bp_1,bp_5]

full_info_ground = Full_Info_Play_Ground(stochastically_constrained_fast_switch_data_job, 
                                                 algorithms,
                                                 plot_label = "Stochastically Constrained (Fast Switch)",
                                                 plot_directory = "results/full_info_simulations/rewards/")
full_info_ground.plot_regret()


bp_1 = BaggingFromThePast_FI(high_gap_stochastic_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = True)
bp_2 = BaggingFromThePast_FI(high_gap_stochastic_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = False)
bp_3 = BaggingFromThePast_FI(high_gap_stochastic_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = True)
bp_4 = BaggingFromThePast_FI(high_gap_stochastic_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = False)
bp_5 = BaggingFromThePast_FI(high_gap_stochastic_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = True)
bp_6 = BaggingFromThePast_FI(high_gap_stochastic_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = False)

student_t = Student_T_FTPL_FI(high_gap_stochastic_data_job)
hedge = Hedge(high_gap_stochastic_data_job)
ftl = Follow_The_Leader_FI(high_gap_stochastic_data_job)

gaussian_ftpl = Gaussian_FTPL_FI(high_gap_stochastic_data_job, num_arms = NUM_ARMS)
gaussian_ftpl_identity_cov = Gaussian_FTPL_FI(high_gap_stochastic_data_job, identity_cov = True, num_arms = NUM_ARMS)
gaussian_ftpl_scaled_down = Gaussian_FTPL_FI(high_gap_stochastic_data_job, scaled_down = True, num_arms = NUM_ARMS)
gaussian_ftpl_identity_cov_scaled_down = Gaussian_FTPL_FI(high_gap_stochastic_data_job, identity_cov = True, scaled_down=True, num_arms = NUM_ARMS)

algorithms = [gaussian_ftpl,
              gaussian_ftpl_identity_cov,
              gaussian_ftpl_scaled_down,
              gaussian_ftpl_identity_cov_scaled_down,
              ftl,
              hedge]


full_info_ground = Full_Info_Play_Ground(high_gap_stochastic_data_job, 
                                                 algorithms,
                                                 plot_label = "Stochastic High Gap", 
                                                 plot_directory = "results/full_info_simulations/rewards/")
full_info_ground.plot_regret()

bp_1 = BaggingFromThePast_FI(low_gap_stochastic_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = True)
bp_2 = BaggingFromThePast_FI(low_gap_stochastic_data_job, get_bag_size = bag_size_t, bag_label = "t", add_bag = False)
bp_3 = BaggingFromThePast_FI(low_gap_stochastic_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = True)
bp_4 = BaggingFromThePast_FI(low_gap_stochastic_data_job, get_bag_size = bag_size_p, bag_label = "t**(0.75)", add_bag = False)
bp_5 = BaggingFromThePast_FI(low_gap_stochastic_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = True)
bp_6 = BaggingFromThePast_FI(low_gap_stochastic_data_job, get_bag_size = bag_size_log, bag_label = "log(t)", add_bag = False)

student_t = Student_T_FTPL_FI(low_gap_stochastic_data_job)
hedge = Hedge(low_gap_stochastic_data_job)
ftl = Follow_The_Leader_FI(low_gap_stochastic_data_job)

gaussian_ftpl = Gaussian_FTPL_FI(low_gap_stochastic_data_job, num_arms = NUM_ARMS)
gaussian_ftpl_identity_cov = Gaussian_FTPL_FI(low_gap_stochastic_data_job, identity_cov = True, num_arms = NUM_ARMS)
gaussian_ftpl_scaled_down = Gaussian_FTPL_FI(low_gap_stochastic_data_job, scaled_down = True, num_arms = NUM_ARMS)
gaussian_ftpl_identity_cov_scaled_down = Gaussian_FTPL_FI(low_gap_stochastic_data_job, identity_cov = True, scaled_down=True, num_arms = NUM_ARMS)

algorithms = [gaussian_ftpl,
              gaussian_ftpl_identity_cov,
              gaussian_ftpl_scaled_down,
              gaussian_ftpl_identity_cov_scaled_down,
              hedge]

full_info_ground = Full_Info_Play_Ground(low_gap_stochastic_data_job, 
                                                 algorithms,
                                                 plot_label = "Stochastic Low Gap",
                                                 plot_directory = "results/full_info_simulations/rewards/")
full_info_ground.plot_regret()
'''
