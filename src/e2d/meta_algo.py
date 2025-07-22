import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import itertools
import math

# This is a Meta-Algorithm
class Meta_Algo():
    def __init__(self, M, K, T, mc_estimator, players, finite_model_class, num_runs, m, title, file_name):
        super().__init__()
        self.M = M # number of models
        self.K = K # number of arms
        self.T = T # time horizon
        self.num_runs = num_runs # number of runs
        self.players = players
        self.model_class = finite_model_class 
        self.accumulated_regret_dec = np.ones(shape = (num_runs, self.T)) # will graph the final regret later on 
        self.accumulated_regret_exp = np.zeros(shape = (num_runs, self.T))
        self.mc_estimator = mc_estimator
        self.gamma = math.sqrt(T)*(100)
        self.m = m
        self.title = title
        self.file_name = file_name

    def compute_averaged_regret(self):
        for i in range(self.num_runs):
            self.mc_estimator.draw_samples(self.m)
            self.f_m_hat = self.mc_estimator.get_f_m_hat()
            self.sq_hellinger_divergence_map = self.mc_estimator.get_sq_hellinger_divergence_hat()

            for j in range(self.T):
                for p in range(len(self.players)):
                    action = self.players[p].select_action(self.f_m_hat, self.sq_hellinger_divergence_map)
                    r_t = self.model_class.get_rt(action)
                    self.players[p].update_training_dataset(r_t, action, self.f_m_hat, self.model_class, i, j)

            self.mc_estimator.clear()

        plt.ylabel('Averaged Regret')
        plt.xlabel('Time Horizon')
        plt.title(self.title)
        for p in range(len(self.players)):
            self.players[p].plot_averaged_regret()
        plt.legend()
        plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/" + self.file_name)
    