import numpy as np
from numpy import random
from e2d.players.player import Player
import matplotlib.pyplot as plt
import math

class Exp3_Player(Player):
    def __init__(self, M, K, T, numRuns, algEst, label):
        super().__init__()
        self.M = M 
        self.K = K 
        self.T = T
        self.algEst = algEst
        self.numRuns = numRuns
        self.accumulated_regret = np.zeros(shape = (self.numRuns, self.T))
        self.label = label

    def select_action(self, f_m_hat, sq_hellinger_div):
        self.algEst.get_m_hat_index()
        exp3_p = self.algEst.current_sampling_distribution()
        p_exp3 = np.zeros(shape = self.K)
        for e in range(self.M):
            p_exp3[np.argmax(f_m_hat[e, :])] += exp3_p[e]
        self.current_p = p_exp3
        return np.random.choice(self.K, p = self.current_p)

    def update_training_dataset(self, r_t, a_t, f_m_hat, model_class, run, t):
        self.algEst.add_to_training_data_set(a_t, r_t, self.current_p[a_t])
        if (t == 0):
            self.accumulated_regret[run, 0] = model_class.compute_instantaneous_regret(self.current_p, self.label, t)
        else:
            self.accumulated_regret[run, t] = self.accumulated_regret[run, t-1] + model_class.compute_instantaneous_regret(self.current_p, self.label, t)

        if (t == self.T - 1):
            self.algEst.clear()
            if (run % 10 == 0):
                print("Final Averaged Regret of Player {0} of run {1} : {2}".format(self.label, run, self.accumulated_regret[run][self.T - 1]))

    def plot_averaged_regret(self):
        sd = 1.96 * (np.std(self.accumulated_regret, axis = 0) / np.sqrt(self.numRuns))
        plt.plot(range(self.T), np.mean(self.accumulated_regret, axis=0), label=self.label)

    def get_final_averaged_regret(self):
        return np.mean(self.accumulated_regret, axis = 0)[self.T - 1]