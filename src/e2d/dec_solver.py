from abc import ABC, abstractmethod
import numpy as np
from numpy import random
from scipy.optimize import linprog

# This is a DEC - LP solver
class DEC_Solver():
    
    # f_m_hat = [|M| x K]
    # hellinger_divergence_hat = [|M| x  K]
    # m_hat is an int in [|M|]
    def __init__(self, M, K, f_m_hat):
        super().__init__()
        self.M = M
        self.K = K
        self.f_m_hat = f_m_hat

    def compute_strategy(self, hellinger_divergence_hat, gamma):
        self.hell_div_hat = hellinger_divergence_hat
    
        # s 
        c = np.zeros(shape = self.K + 1)
        c[0] = 1
        
        # b refers to the |M| inequality constraints 
        A_ub = np.zeros(shape = (self.M, self.K + 1))
        b = np.zeros(shape = self.M)
        for i in range(self.M):
            b[i] = -1 * np.max(self.f_m_hat[i, :])
            A_ub[i][0] = -1 
            for j in range(self.K):
                A_ub[i][j + 1] = (-1 * (gamma * self.hell_div_hat[i][j]) + self.f_m_hat[i][j])

        # ensuring that p is in the simplex
        A_eq = np.ones((self.K + 1))
        A_eq[0] = 0
        b_eq = 1

        # bounds 
        bounds = [(0, None)]
        for i in range(self.K):
            bounds.append((0, 1))
        
        res = linprog(c, A_ub = A_ub, b_ub = b, 
                      A_eq = A_eq, b_eq = b_eq, bounds = bounds)
        response = res.x
        
        self.DEC_hat = response[0]
        self.p_hat = response[1:]
