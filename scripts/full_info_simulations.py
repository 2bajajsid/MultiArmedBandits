
import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from data_generating_mechanism.low_rank_projection import Low_Rank_Stochastic
from data_generating_mechanism.high_gap_stochastic import High_Gap_Stochastic
from play_ground.full_info_play_ground import Full_Info_Play_Ground
from bandit_algorithms.hedge import Hedge
from bandit_algorithms.gaussian_ftpl import Gaussian_FTPL_FI
from bandit_algorithms.follow_the_leader import Follow_The_Leader_FI
import math
import numpy as np
from threading import Thread
import multiprocessing as mp

NUM_ARMS = 15
MONTE_CARLO_RUNS = 65

def target_func(data_job, plot_label):
    algorithms = []
    for i in range(3):
        for j in range(3):
            for k in range(2):
                algorithms.append(Gaussian_FTPL_FI(data_job, 
                                                   num_arms=NUM_ARMS, 
                                                   mean_scale=i,
                                                   covariance_scale=j,
                                                   covariance_type=k))
                
    algorithms.append(Hedge(data_job))
    algorithms.append(Follow_The_Leader_FI(data_job))

    full_info_ground = Full_Info_Play_Ground(data_job, 
                                         algorithms, 
                                         plot_label = plot_label,
                                         plot_directory="rewards")
    full_info_ground.plot_regret(top_n = 4)

data_jobs = [Low_Rank_Stochastic(num_arms = NUM_ARMS, low_rank_dimension=2, 
                                 M = MONTE_CARLO_RUNS, optimality_gap=0.03),
             Low_Rank_Stochastic(num_arms = NUM_ARMS, low_rank_dimension=5, 
                                 M = MONTE_CARLO_RUNS, optimality_gap=0.03),
             Low_Rank_Stochastic(num_arms = NUM_ARMS, low_rank_dimension=2, 
                                 M = MONTE_CARLO_RUNS, optimality_gap=0),
             Low_Rank_Stochastic(num_arms = NUM_ARMS, low_rank_dimension=5, 
                                 M = MONTE_CARLO_RUNS, optimality_gap=0),
             High_Gap_Stochastic(num_arms=NUM_ARMS, M = MONTE_CARLO_RUNS)]

plot_labels = ["Low_Rank_Projection_dimension_2_Gap_point_zero_three",
               "Low_Rank_Projection_dimension_5_Gap_point_zero_three",
               "Low_Rank_Projection_dimension_2_Gap_zero",
               "Low_Rank_Projection_dimension_5_Gap_zero",
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
