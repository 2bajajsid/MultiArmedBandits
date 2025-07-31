import numpy as np
from numpy import random
from e2d.players.player import Player
import itertools
import math
from e2d.technical_tools.dec_solver import DEC_Solver
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

        sampling_distr = self.algEst.current_sampling_distribution()
        m_t = np.random.choice(self.M, p = sampling_distr)
        sampling_distr = np.zeros(shape = self.M)
        sampling_distr[m_t] = 1

        self.dec_solver.compute_strategy(f_m_hat, sq_hellinger_map, self.gamma, m_hat_index, sampling_distr)
        self.current_p = self.dec_solver.p_hat / np.sum(self.dec_solver.p_hat)
        self.dec = self.dec_solver.DEC_hat
        if (self.gamma == 0.5):
            print(self.current_p)
            print("DEC {} for gamma {}".format(self.dec, self.gamma))
        return np.random.choice(self.K, p = self.current_p)

    def update_training_dataset(self, r_t, a_t, f_m_hat, model_class, run, t):
        self.algEst.add_to_training_data_set(a_t, r_t, self.current_p[a_t])
        if (t == 0):
            self.accumulated_regret[run][0] = model_class.compute_instantaneous_regret(self.current_p, self.label, t)
        else:
            self.accumulated_regret[run][t] = self.accumulated_regret[run][t-1] + model_class.compute_instantaneous_regret(self.current_p, self.label, t)

        if (t == self.T - 1):
            #print("Clearing")
            self.algEst.clear()
            if (run % 10 == 0):
                print("Regret of Player {0} of run {1} : {2} with DEC Estimate {3}".format(self.label, run, 
                                                                                           self.accumulated_regret[run][self.T - 1], self.dec))

    def plot_averaged_regret(self):
        sd = 1.96 * (np.std(self.accumulated_regret, axis = 0) / np.sqrt(self.numRuns))
        plt.plot(range(self.T), np.mean(self.accumulated_regret, axis=0), label=self.label)

    def get_final_averaged_regret(self):
        return np.mean(self.accumulated_regret, axis = 0)[self.T - 1]