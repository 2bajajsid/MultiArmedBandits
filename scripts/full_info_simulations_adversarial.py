
import sys
sys.path.append('/Users/sidbajaj/MultiArmedBandits/src')

import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from data_generating_mechanism.low_gap_stochastic import Low_Gap_Stochastic
from data_generating_mechanism.hadamard_adversarial import Hadamard_Adversarial
from data_generating_mechanism.bernoulli_adversarial import Bernoulli_Adversarial
from data_generating_mechanism.fixed_low_rank_exp_3 import Fixed_Low_Rank_Exp_3
from data_generating_mechanism.flip_flop_mechanism import Flip_Flop_Mechanism
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

NUM_ARMS = 8
TIME_HORIZON = 500
INIT_EXPLORATION = 0
MONTE_CARLO_RUNS = 150

def target_func(data_job, plot_label):
    gaussian_ftpl = Gaussian_FTPL_FI(data_job, num_arms = NUM_ARMS)
    gaussian_ftpl_identity_cov = Gaussian_FTPL_FI(data_job, identity_cov = True, num_arms = NUM_ARMS)
    gaussian_ftpl_scaled_down = Gaussian_FTPL_FI(data_job, scaled_down = True, num_arms = NUM_ARMS)
    gaussian_ftpl_zero_mean_scaled_down = Gaussian_FTPL_FI(data_job, scaled_down = True, zero_mean = True, num_arms = NUM_ARMS)
    gaussian_ftpl_identity_cov_scaled_down = Gaussian_FTPL_FI(data_job, identity_cov = True, scaled_down=True, num_arms = NUM_ARMS)
    gaussian_ftpl_identity_cov_zero_mean_scaled_down = Gaussian_FTPL_FI(data_job, scaled_down = True, identity_cov = True, zero_mean=True, num_arms = NUM_ARMS)
    hedge = Hedge(data_job)

    algorithms = [
        gaussian_ftpl,
        gaussian_ftpl_scaled_down,
        gaussian_ftpl_zero_mean_scaled_down,
        gaussian_ftpl_identity_cov,
        gaussian_ftpl_identity_cov_scaled_down,
        gaussian_ftpl_identity_cov_zero_mean_scaled_down,
        hedge
    ]

    full_info_ground = Full_Info_Play_Ground(data_job, 
                                         algorithms, 
                                         plot_label = plot_label,
                                         plot_directory="rewards")
    full_info_ground.plot_regret()

data_jobs = [Hadamard_Adversarial(M = MONTE_CARLO_RUNS)]
plot_labels = ["Hadamard Adversarial"]

'''
data_jobs = [Fixed_Low_Rank_Exp_3(M = MONTE_CARLO_RUNS),
             Hadamard_Adversarial(M = MONTE_CARLO_RUNS),
             Bernoulli_Adversarial(M = MONTE_CARLO_RUNS),
             Low_Gap_Stochastic(num_arms=NUM_ARMS, 
                                M = MONTE_CARLO_RUNS),
             Flip_Flop_Mechanism(num_arms=8, 
                                 M = MONTE_CARLO_RUNS)]

plot_labels = ["Latent Factor - Binary Loadings (Low Gap)",
               "Hadamard Adversarial",
               "Bernoulli Adversarial", 
               "Low Gap Stochastic",
               "Flip Flop Mechanism"]
'''

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