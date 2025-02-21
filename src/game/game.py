from abc import ABC, abstractmethod
import numpy as np
from numpy import random

class Game:
    def __init__(self, data_generating_mechanism, bandit_algorithm):
        self.data_generating_mechanism = data_generating_mechanism
        self.bandit_algorithm = bandit_algorithm
        self.accumulated_regret = np.zeros(shape=(data_generating_mechanism.get_M(), 
                                                  data_generating_mechanism.get_T()))
        self.label = self.bandit_algorithm.label
        
    @abstractmethod
    def simulate_one_run(self):
        pass

    def compute_averaged_regret(self):
        for i in range(self.data_generating_mechanism.get_M()):
            self.accumulated_regret[i, :] = self.simulate_one_run()
            print('m = {:d}'.format(i))
        print('Regret of bandit algorithm {} over {:d} runs calculated'.format(self.label, self.data_generating_mechanism.get_M()))

    def get_averaged_regret(self):
        return np.mean(self.accumulated_regret, axis=0)