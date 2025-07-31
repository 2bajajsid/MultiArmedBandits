import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import itertools
import math
from e2d.model_collection.gaussian_model_collection import Gaussian_Model_Collection
from e2d.technical_tools.mc_estimator import MC_Estimator
from e2d.players.exp3_player import Exp3_Player
from e2d.players.dec_player import DEC_Player
from e2d.oracle.exp_weights_oracle import Exp_Weights_Oracle

# This is a Meta-Algorithm
class Meta_Algo():
    def __init__(self, M, K, T, num_runs, title, file_name, optimality_gap, delta):
        super().__init__()
        self.M = M # number of models
        self.K = K # number of arms
        self.T = T # time horizon
        self.num_runs = num_runs # number of runs
        self.accumulated_regret_dec = np.ones(shape = (num_runs, self.T)) # will graph the final regret later on 
        self.accumulated_regret_exp = np.zeros(shape = (num_runs, self.T))
        self.optimality_gap = optimality_gap
        self.delta = delta
        self.gamma = math.sqrt(T)*(100)
        self.title = title
        self.file_name = file_name

    def compute_averaged_regret(self):
        self.players = []
        self.players.append(Exp3_Player(M = self.M, T = self.T, K = self.K, numRuns = self.num_runs, 
                                        algEst=Exp_Weights_Oracle(T = self.T, M = self.M, K = self.K), label="EXP3"))
        
        gamma = [0.1, 1.0, 5.0, 10.0, 500.0, 1000.0]
        for g in range(len(gamma)):
            self.players.append(DEC_Player(M = self.M, T = self.T, K = self.K, numRuns = self.num_runs, 
                                            algEst=Exp_Weights_Oracle(T = self.T, M = self.M, K =self.K), 
                            gamma = gamma[g], label="DEC (gamma): {}".format(gamma[g])))
                
        for i in range(self.num_runs):
            self.model_class = Gaussian_Model_Collection(K = self.K, M = self.M, Optimality_Gap=self.optimality_gap)
            self.mc_estimator = MC_Estimator(finite_model_class = self.model_class)
            self.m = self.mc_estimator.get_m_for_gaussian_model(delta = self.delta)
            self.mc_estimator.draw_samples(self.m)
            self.f_m_hat = self.mc_estimator.get_f_m_hat()
            
            self.sq_hellinger_divergence_matrix = self.mc_estimator.get_sq_hellinger_divergence_hat()
            self.sq_hellinger_divergence_map = np.zeros(shape = (self.M, self.M, self.K))

            for p in range(len(self.players)):
                self.players[p].algEst.set_model_class(self.model_class)

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

        print("*******")

        sorted_players = np.argsort(players_regret)
        for i in range(3):
            player_i = self.players[sorted_players[i]]
            player_i.plot_averaged_regret()
            print("Player {} has final averaged regret {} with std {}".format(player_i.label, 
                                                                              np.mean(player_i.accumulated_regret, axis = 0)[self.T - 1], 
                                                                              np.std(player_i.accumulated_regret, axis = 0)[self.T - 1] / 
                                                                              np.sqrt(player_i.numRuns)))
        
        plt.legend()
        plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/" + self.file_name)
        print("Stats for game with gap {0} and probability {1}".format(self.optimality_gap, self.delta))
    
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
    