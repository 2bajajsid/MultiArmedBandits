import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI, Bandit_Algorithm_FI
import math

class BaggingFromThePast_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism, get_bag_size, bag_label, add_bag):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.get_bag_size = get_bag_size
        self.add_bag = add_bag

        self.__data_generating_mechanism = data_generating_mechanism
        self.__label = "BP ({} bag size {})".format("adding" if add_bag else "subtracting", 
                                                    bag_label)

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t):
        if (t < self.exploration_phase_length):
            return math.floor(t / self.init_exploration)
        else:
            copied_history = np.copy(losses)
            uniform_sample = list(random.choice(t, size = self.get_bag_size(t), replace=False))
                
            for i in range(len(uniform_sample)):
                copied_history[:, uniform_sample[i]] = -1 * copied_history[:, uniform_sample[i]]

            arm_estimates_current_round = np.mean(copied_history, axis = 1)
            return np.argmin(arm_estimates_current_round)
        
class BaggingFromThePast_PI(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism, get_bag_size, bag_label, add_bag):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.num_bootstrap_simulations = 100
        self.get_bag_size = get_bag_size
        self.add_bag = add_bag

        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "BP ({} bag size {})".format("adding" if add_bag else "subtracting", 
                                                    bag_label)
        
        print(self.__label)

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label
    
    @property
    def current_sampling_distribution(self):
        return self.__current_sampling_distribution
    
    @current_sampling_distribution.setter
    def current_sampling_distribution(self, distr):
        self.__current_sampling_distribution = distr

    def get_arm_to_pull(self, importance_weighted_losses, losses, t):
        if (t < self.exploration_phase_length):
            A_t = math.floor(t / self.data_generating_mechanism.get_init_exploration())
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
        else:
            num_counts = np.zeros(shape = self.K)
            
            for n in range(self.num_bootstrap_simulations):
                uniform_sample = list(random.choice(t, size = self.get_bag_size(t), replace=False))
                arm_estimates_current_simulation = np.sum(importance_weighted_losses, axis = 1)

                if (self.add_bag):
                    arm_estimates_current_simulation += np.sum(importance_weighted_losses[:, uniform_sample], axis = 1)
                else: 
                    arm_estimates_current_simulation -= np.sum(importance_weighted_losses[:, uniform_sample], axis = 1)

                arm_chosen_this_simulation = np.argmin(arm_estimates_current_simulation)
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1
                if (self.num_bootstrap_simulations == n + 1):
                    A_t = arm_chosen_this_simulation
                
            p_t = num_counts / sum(num_counts)
        
            # flip a bernoulli with success 
            # probability of 1/t to add an 
            # explicit exploration component
            z = np.random.binomial(n = 1, p = (1 / t))
            self.current_sampling_distribution = ((1 - (1 / t)) * p_t) + (1 / (t * self.K))

            if (z == 1):
                A_t = random.randint(low = 0, high = self.K, size = 1)
        
        return A_t