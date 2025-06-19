from abc import ABC, abstractmethod
import numpy as np
from numpy import random

class Game:
    def __init__(self, bandit_algorithm):
        self.bandit_algorithm = bandit_algorithm
        self.data_generating_mechanism = self.bandit_algorithm.data_generating_mechanism
        self.accumulated_regret = np.zeros(shape=(self.data_generating_mechanism.get_M(), 
                                                  self.data_generating_mechanism.get_T()))
        self.label = self.bandit_algorithm.label
        
    @abstractmethod
    def simulate_one_run(self):
        pass

    def compute_averaged_regret(self, hyperparameter):
        for i in range(self.data_generating_mechanism.get_M()):
            self.accumulated_regret[i, :] = self.simulate_one_run(hyperparameter)
            #if i % 500 == 0:
            #    print('m = {:d}'.format(i))
        print('Average Regret of delta {} over {} runs calculated (median: {} mean: {} std: {})'
              .format(self.data_generating_mechanism.delta, 
                      self.data_generating_mechanism.get_M(),
                      np.median(self.accumulated_regret, axis=0)[self.data_generating_mechanism.get_T() - 1],
                      np.mean(self.accumulated_regret, axis=0)[self.data_generating_mechanism.get_T() - 1], 
                      np.std(self.get_regret_final()) / np.sqrt(self.data_generating_mechanism.get_M())))

    def get_averaged_regret(self):
        return np.mean(self.accumulated_regret, axis=0)
    
    def get_median_regret(self):
        return np.median(self.accumulated_regret, axis=0)
    
    def get_regret_final(self):
        return self.accumulated_regret[:, self.data_generating_mechanism.get_T() - 1]
    
    def get_instantaneuous_regret(self, hyperparameter):
        self.compute_averaged_regret(hyperparameter)
        ave_regret = self.get_averaged_regret()
        instantaneuous_regret = np.zeros(shape = self.data_generating_mechanism.get_T())

        for i in range(self.data_generating_mechanism.get_T()):
            if (i == 0):
                instantaneuous_regret[i] = ave_regret[i]
            else:
                instantaneuous_regret[i] = (ave_regret[i] - ave_regret[i-1])

        print("Average Regret of {}".format(self.bandit_algorithm.label))
        print(ave_regret[self.data_generating_mechanism.get_T() - 1])

        return instantaneuous_regret

