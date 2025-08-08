import numpy as np
from numpy import random
import matplotlib.pyplot as plt
import itertools
import math
from e2d.model_collection.gaussian_model_collection import Gaussian_Model_Collection
from e2d.model_collection.bernoulli_model_collection import Bernoulli_Model_Collection
from e2d.technical_tools.mc_estimator import MC_Estimator
from e2d.players.exp3_player import Exp3_Player
from e2d.players.dec_player import DEC_Player
from e2d.oracle.exp_weights_oracle import Exp_Weights_Oracle
from e2d.technical_tools.constants import HOEFFDING_SAMPLE_SIZE, GAUSSIAN_MODELS, BERNOULLI_MODELS

# This is a Meta-Algorithm
class Meta_Algo():
    def __init__(self, M, K, T, num_runs, title, file_name, optimality_gap, delta, sample_size_type, divergence_type, type_model):
        super().__init__()
        self.M = M # number of models
        self.K = K # number of arms
        self.T = T # time horizon
        self.num_runs = num_runs # number of runs
        self.accumulated_regret_dec = np.ones(shape = (num_runs, self.T)) # will graph the final regret later on 
        self.accumulated_regret_exp = np.zeros(shape = (num_runs, self.T))
        self.optimality_gap = optimality_gap
        self.delta = delta
        self.title = title
        self.file_name = file_name
        self.sample_size_type = sample_size_type
        self.divergence_type = divergence_type
        self.model_type = type_model
        #self.model_class = Gaussian_Model_Collection(K = self.K, M = self.M, Optimality_Gap=self.optimality_gap)
        #self.model_class = Bernoulli_Model_Collection(K = self.K, M = self.M, Optimality_Gap=self.optimality_gap)
        #self.mc_estimator = MC_Estimator(finite_model_class = self.model_class)

    def compute_averaged_regret(self):
        self.players = []
        #self.players.append(Exp3_Player(M = self.M, T = self.T, K = self.K, numRuns = self.num_runs, 
        #                                algEst=Exp_Weights_Oracle(T = self.T, M = self.M, K = self.K), label="EXP3"))
        
        gamma =  [0.1, 1.0, 5.0, 10.0, 500.0, 1000.0]
        for g in range(len(gamma)):
            self.players.append(DEC_Player(M = self.M, T = self.T, K = self.K, numRuns = self.num_runs, 
                                            algEst=Exp_Weights_Oracle(T = self.T, M = self.M, K =self.K), 
                            gamma = -1 * gamma[g], label="sample_size_type: {}, gamma: {}".format(self.sample_size_type, 
                                                                                             gamma[g])))
                
        for i in range(self.num_runs):
            if (self.model_type == GAUSSIAN_MODELS):
                self.model_class = Gaussian_Model_Collection(K = self.K, M = self.M, Optimality_Gap=self.optimality_gap)
            else:
                self.model_class = Bernoulli_Model_Collection(K = self.K, M = self.M, Optimality_Gap=self.optimality_gap)
            self.mc_estimator = MC_Estimator(finite_model_class = self.model_class)

            if (self.sample_size_type >= HOEFFDING_SAMPLE_SIZE):
                self.mc_estimator.compute_sample_size_m(delta = self.delta, type = self.sample_size_type)
            
            self.f_m_hat = self.mc_estimator.get_f_m_hat(self.sample_size_type)
            self.sq_hellinger_divergence_matrix = self.mc_estimator.get_divergence_hat(sample_size_type=self.sample_size_type,
                                                                                       divergence_type=self.divergence_type)
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

            self.mc_estimator.clear(self.sample_size_type)

        #plt.ylabel('Averaged Regret')
        #plt.xlabel('Time Horizon')
        #plt.title(self.title)

        players_regret = np.zeros(shape = len(self.players))
        for p in range(len(self.players)):
            players_regret[p] = self.players[p].get_final_averaged_regret()

        #print("*******")

        sorted_players = np.argsort(players_regret)
        #for i in range(3):
        #    player_i = self.players[sorted_players[i]]
        #    player_i.plot_averaged_regret()
        #    print("Player {} has final averaged regret {} with std {}".format(player_i.label, 
        #                                                                      np.mean(player_i.accumulated_regret, axis = 0)[self.T - 1], 
        #                                                                     np.std(player_i.accumulated_regret, axis = 0)[self.T - 1] / 
        #                                                                     np.sqrt(player_i.numRuns)))
        
        #plt.legend()
        #plt.savefig("/Users/sidbajaj/MultiArmedBandits/results/e2d/" + self.file_name)
        #print("Stats for game with gap {0} and probability {1}".format(self.optimality_gap, self.delta))

        best_player = self.players[sorted_players[0]]
        return [np.mean(best_player.accumulated_regret, axis = 0), 
                1.96 * (np.std(best_player.accumulated_regret, axis = 0) / np.sqrt(self.T)), 
                best_player.label,
                best_player.gamma]
    
    def get_true_radon_nikodym_derivatives(self, dp_or_dq = 1, model_index_1 = 0, model_index_2 = 1, action = 0):
        self.mc_estimator = MC_Estimator(finite_model_class = self.model_class)
        x = np.linspace(-3, 3, num = 50)
        y = np.zeros(shape = 50)
        for i in range(50):
            y[i] =  self.model_class.compute_true_radon_nikodym_derivative(model_index_1, model_index_2, x[i])[dp_or_dq][action]
        return [x, y]
    
    def get_estimated_radon_nikodym_derivatives(self, m, dp_or_dq = 1, model_index_1 = 0, model_index_2 = 1, action = 0):
        self.mc_estimator = MC_Estimator(finite_model_class = self.model_class)
        self.mc_estimator.m = m
        
        x = np.linspace(-3, 3, num = 50)
        y = self.mc_estimator.estimated_radon_nikodym_derivative(model_index_i=model_index_1,
                                                                            model_index_j=model_index_2,
                                                                            x = x)[dp_or_dq][action]
        return [x, y]
    
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
    