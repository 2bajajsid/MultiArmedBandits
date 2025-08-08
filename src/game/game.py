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
        print("Simulating with parameter {}".format(hyperparameter))
        for i in range(self.data_generating_mechanism.get_M()):
            self.accumulated_regret[i, :] = self.simulate_one_run(hyperparameter)
            if i % 1000 == 0:
                print(hyperparameter)
                print('m = {:d}'.format(i))
                print(self.accumulated_regret[i, 999])
            
        self.compute_regret_sub()    
        print('Average Regret of delta {} over {} runs calculated (median: {} mean: {} std: {})'
              .format(self.data_generating_mechanism.delta, 
                      self.data_generating_mechanism.prior_samples,
                      np.median(self.regret_sub_mean),
                      np.mean(self.regret_sub_mean), 
                      np.std(self.regret_sub_mean) / np.sqrt(self.data_generating_mechanism.prior_samples)))
        
        return np.mean(self.regret_sub_mean)
        
    def compute_regret_sub(self):
        self.regret_final = self.get_regret_final()
        self.regret_sub_mean = np.zeros(self.data_generating_mechanism.prior_samples)
        self.accumulated_regret_sub = np.zeros(shape=(self.data_generating_mechanism.prior_samples, self.data_generating_mechanism.get_T()))
        for i in range(self.data_generating_mechanism.prior_samples):
            start_ind = i*self.data_generating_mechanism.prior_repeats
            end_ind = (i+1)*self.data_generating_mechanism.prior_repeats - 1
            self.regret_sub_mean[i] = np.mean(self.regret_final[start_ind:end_ind])
            self.accumulated_regret_sub[i,:] = np.mean(self.accumulated_regret[start_ind:end_ind,:], axis=0)
        print(self.regret_sub_mean)
        print("Quantiles (25, 50, 75, 90, 99)")
        print(np.quantile(self.regret_sub_mean, [0.25, 0.5, 0.75, 0.9, 0.99]))

    def get_sd_final(self):
        return 1.96 * (np.std(self.regret_sub_mean) / np.sqrt(self.data_generating_mechanism.prior_samples))

    def get_median_final_regret(self):
        return np.median(self.regret_sub_mean)

    def get_averaged_regret(self):
        return np.mean(self.accumulated_regret_sub, axis=0)
    
    def get_median_regret(self):
        return np.median(self.accumulated_regret_sub, axis=0)
    
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

