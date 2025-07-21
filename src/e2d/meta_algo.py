import numpy as np
from numpy import random
from e2d.dec_solver import DEC_Solver
import matplotlib.pyplot as plt
import itertools
import math

# This is a Meta-Algorithm
class Meta_Algo():
    def __init__(self, M, K, T, mc_estimator, alg_est, finite_model_class, num_runs, m):
        super().__init__()
        self.M = M # number of models
        self.K = K # number of arms
        self.T = T # time horizon
        self.num_runs = num_runs # number of runs
        self.oracle = alg_est # exp3 
        self.model_class = finite_model_class 
        self.accumulated_regret_dec = np.zeros(shape = (num_runs, self.T)) # will graph the final regret later on 
        self.accumulated_regret_exp = np.zeros(shape = (num_runs, self.T))
        self.mc_estimator = mc_estimator
        self.gamma = math.sqrt(T)
        self.m = m

    def compute_averaged_regret(self):
        for i in range(self.num_runs):
            self.mc_estimator.draw_samples(self.m)
            self.f_m_hat = self.mc_estimator.get_f_m_hat()
            self.sq_hellinger_divergence_map = self.mc_estimator.get_sq_hellinger_divergence_hat()
            self.dec_solver = DEC_Solver(self.M, self.K, self.f_m_hat)

            for j in range(self.T):
                # estimation ... 
                m_hat_index = self.oracle.get_m_hat_index()
                exp3_p = self.oracle.current_sampling_distribution()
                # sq_hellinger_map = self.get_sq_hellinger_map(m_hat_index)
                
                # to decision ....
                # self.dec_solver.compute_strategy(sq_hellinger_map, self.gamma)
                # action = np.random.choice(self.M, p = self.dec_solver.p_hat)
                p_exp3 = np.zeros(shape = self.K)
                for e in range(self.M):
                    p_exp3[np.argmax(self.f_m_hat[e, :])] += exp3_p[e]
                action = np.random.choice(self.K, p = p_exp3)
                o_t = self.model_class.get_ot(action)

                if (j == 0):
                    # self.accumulated_regret_dec[i][0] = self.finite_model_class.compute_instantaneuous_regret(self.exp3_p)
                    self.accumulated_regret_exp[i][0] = self.model_class.compute_instantaneous_regret(p_exp3)
                else:
                    # self.accumulated_regret_dec[i][j] = self.accumulated_regret[i][j-1] + self.finite_model_class.compute_instantaneous_regret(self.dec_solver.p_hat)
                    self.accumulated_regret_exp[i][j] = self.accumulated_regret_exp[i][j-1] + self.model_class.compute_instantaneous_regret(p_exp3)
                self.oracle.add_to_training_data_set(o_t)

            print("Final Averaged Regret of run {0} : {1}".format(i, self.accumulated_regret_exp[i][j]))

            self.oracle.clear()
            self.mc_estimator.clear()

        plt.ylabel('Averaged Regret (Exp3)')
        plt.xlabel('Time Horizon')
        plt.title('Averaged Regret over Time')
        plt.plot(range(self.T), np.mean(self.accumulated_regret_exp, axis=0))
        plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/AveragedRegret")

    def get_sq_hellinger_map(self, m_hat_index):
        combs = itertools.combinations(range(self.M), 2)
        relevant_comb_indices = []

        searching_for_model_ind = 0
        if (m_hat_index == 0):
            searching_for_model_ind = 1

        for i in range(len(combs)):
            if ((combs[0] == searching_for_model_ind and combs[1] == m_hat_index) or 
                (combs[0] == m_hat_index and combs[1] == searching_for_model_ind)):
                relevant_comb_indices.append(i)
                searching_for_model_ind += 1
    