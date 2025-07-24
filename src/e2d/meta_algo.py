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
            
            self.sq_hellinger_divergence_matrix = self.mc_estimator.get_sq_hellinger_divergence_hat()
            self.sq_hellinger_divergence_map = np.zeros(shape = (self.M, self.M, self.K))
            for y in range(self.M):
                self.sq_hellinger_divergence_map[y] = self.get_sq_hellinger_map(y, self.sq_hellinger_divergence_matrix)

            for j in range(self.T):
                for p in range(len(self.players)):
                    action = self.players[p].select_action(self.f_m_hat, self.sq_hellinger_divergence_map)
                    r_t = self.model_class.get_rt(action)
                    self.players[p].update_training_dataset(r_t, action, self.f_m_hat, self.model_class, i, j)

            self.mc_estimator.clear()

        plt.ylabel('Averaged Regret')
        plt.xlabel('Time Horizon')
        plt.title(self.title)

        players_regret = np.zeros(shape = len(self.players))
        for p in range(len(self.players)):
            players_regret[p] = self.players[p].get_final_averaged_regret()

        sorted_players = np.argsort(players_regret)
        for i in range(3):
            self.players[sorted_players[i]].plot_averaged_regret()

        plt.legend()
        plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/" + self.file_name)
    
    def get_sq_hellinger_map(self, m_hat_index, sq_hellinger_divergence_map):
        combs = list(itertools.combinations(range(self.M), 2))
        relevant_comb_indices = []

        searching_for_model_ind = 0
        if (m_hat_index == 0):
            searching_for_model_ind = 1

        for i in range(len(combs)):
            if ((combs[i][0] == m_hat_index and combs[i][1] == searching_for_model_ind)):
                relevant_comb_indices.append(i)
                searching_for_model_ind += 1
            elif((combs[i][1] == m_hat_index and combs[i][0] == searching_for_model_ind)):
                relevant_comb_indices.append(i)
                searching_for_model_ind += 1
                if (searching_for_model_ind == m_hat_index):
                    searching_for_model_ind += 1
        sq_hellinger_map = np.zeros(shape = (self.M, self.K))
        j = 0
        for i in range(self.M):
            if (i != m_hat_index):
                sq_hellinger_map[i, :] = sq_hellinger_divergence_map[relevant_comb_indices[j], :]
                j += 1
        return sq_hellinger_map
    