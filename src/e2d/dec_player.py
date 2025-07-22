import numpy as np
from numpy import random
from e2d.player import Player
import itertools
import math
from e2d.dec_solver import DEC_Solver
import matplotlib.pyplot as plt

class DEC_Player(Player):
    def __init__(self, M, K, T, numRuns, algEst, label, gamma):
        super().__init__()
        self.M = M 
        self.K = K 
        self.T = T
        self.algEst = algEst
        self.numRuns = numRuns
        self.accumulated_regret = np.zeros(shape = (self.numRuns, self.T))
        self.label = label
        self.gamma = gamma
        self.dec_solver = DEC_Solver(M = M, K = K)

    def select_action(self, f_m_hat, sq_hellinger_map):
        m_hat_index = self.algEst.get_m_hat_index()
        sq_hellinger_map = self.get_sq_hellinger_map(m_hat_index, sq_hellinger_map)
        self.dec_solver.compute_strategy(f_m_hat, sq_hellinger_map, self.gamma)
        self.current_p = self.dec_solver.p_hat
        return np.random.choice(self.K, p = self.current_p)

    def update_training_dataset(self, r_t, a_t, f_m_hat, model_class, run, t):
        self.algEst.add_to_training_data_set(f_m_hat[:, a_t], r_t)
        if (t == 0):
            self.accumulated_regret[run][0] = model_class.compute_instantaneous_regret(self.current_p)
        else:
            self.accumulated_regret[run][t] = self.accumulated_regret[run][t-1] + model_class.compute_instantaneous_regret(self.current_p)

        if (t == self.T - 1):
            self.algEst.clear()
            print("Final Averaged Regret of Player {0} of run {1} : {2}".format(self.label, run, self.accumulated_regret[run][self.T - 1]))

    def plot_averaged_regret(self):
        sd = 1.96 * (np.std(self.accumulated_regret, axis = 0) / np.sqrt(self.numRuns))
        plt.plot(range(self.T), np.mean(self.accumulated_regret, axis=0), label=self.label)

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