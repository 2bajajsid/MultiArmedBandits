import numpy as np
from numpy import random
from scipy.optimize import linprog
from e2d.dec_solver import DEC_Solver
import itertools
import math

# This is a Meta-Algorithm
class Meta_Algo():
    def __init__(self, M, K, T, mc_estimator, alg_est, finite_model_class, num_runs = 1):
        super().__init__()
        self.M = M # number of models
        self.K = K # number of arms
        self.T = T # time horizon
        self.num_runs = num_runs # number of runs
        self.oracle = alg_est # exp3 
        self.model_class = finite_model_class 
        self.accumulated_regret = np.zeros(shape = (num_runs, self.T)) # will graph the final regret later on 
        self.mc_estimator = mc_estimator
        self.gamma = math.sqrt(T)

    def compute_averaged_regret(self):
        for i in range(self.num_runs):
            self.mc_estimator.draw_samples()
            self.f_m_hat = self.mc_estimator.get_f_m_hat()
            self.sq_hellinger_divergence_map = self.mc_estimator.get_sq_hellinger_divergence_hat()
            self.dec_solver = DEC_Solver(self.M, self.K, self.f_m_hat)

            for j in range(self.T):
                # estimation ... 
                m_hat_index = self.oracle.get_m_hat_index()
                sq_hellinger_map = self.get_sq_hellinger_map(m_hat_index)
                
                # to decision ....
                self.dec_solver.compute_strategy(sq_hellinger_map, self.gamma)
                action = np.random.choice(self.M, p = self.dec_solver.p_hat)
                
                o_t = self.finite_model_class.get_ot(action)
                if (j > 0):
                    self.accumulated_regret[i][j] = self.accumulated_regret[i][j-1] + self.finite_model_class.compute_instantaneous_regret(self.dec_solver.p_hat)
                self.oracle.add_to_training_dataset(o_t, o_t[action])

            self.oracle.clear()
            self.mc_estimator.clear()

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
    