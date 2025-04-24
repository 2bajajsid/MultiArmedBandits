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
            cumulative_losses = np.sum(losses, axis = 1)
            uniform_sample = list(random.choice(t, size = self.get_bag_size(t), replace=True))
            bag_losses = np.sum(losses[:, uniform_sample], axis=1)
                
            if (self.add_bag):
                cumulative_losses += bag_losses
            else:
                cumulative_losses -= bag_losses

            return np.argmin(cumulative_losses)
        
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
            A_t = math.floor(t / self.init_exploration)
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
        else:
            num_counts = np.zeros(shape = self.K)
            
            for n in range(self.num_bootstrap_simulations):
                uniform_sample = list(random.choice(t, size = self.get_bag_size(t), replace=True))
                bag_losses = np.sum(importance_weighted_losses[:, uniform_sample], axis=1)
                cumulative_losses = np.sum(importance_weighted_losses, axis = 1)

                if (self.add_bag):
                    cumulative_losses += bag_losses
                else: 
                    cumulative_losses -= bag_losses

                arm_chosen_this_simulation = np.argmin(cumulative_losses)
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1
                if (n == self.num_bootstrap_simulations - 1):
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
    
class BaggingFromThePast_Cache_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism, get_bag_size, bag_label, add_bag):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.get_bag_size = get_bag_size
        self.add_bag = add_bag
        self.bootstrap_loss_estimates = np.zeros(shape = self.K)
        self.a_t_minus_1 = -1

        self.__data_generating_mechanism = data_generating_mechanism
        self.__label = "Cached BP ({} bag size {})".format("adding" if add_bag else "subtracting", 
                                                    bag_label)

    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t, historyIsRewards = False):
        if (t < self.exploration_phase_length):
            return math.floor(t / self.init_exploration)
        else:
            uniform_sample = list(random.choice(t, size = self.get_bag_size(t), replace=True))
            if (self.a_t_minus_1 == -1):
                cumulative_losses = np.sum(losses, axis = 1)
                bag_losses = np.sum(losses[:, uniform_sample], axis=1)
                self.bootstrap_loss_estimates = (cumulative_losses + bag_losses) if self.add_bag else (cumulative_losses - bag_losses)
            else:
                cumulative_losses = np.sum(losses[self.a_t_minus_1, :])
                bag_losses = np.sum(losses[self.a_t_minus_1, uniform_sample])
                self.bootstrap_loss_estimates[self.a_t_minus_1] = (cumulative_losses + bag_losses) if self.add_bag else (cumulative_losses - bag_losses)
                
            self.a_t_minus_1 = np.argmax(self.bootstrap_loss_estimates) if historyIsRewards else np.argmin(self.bootstrap_loss_estimates)
            return self.a_t_minus_1
            
    
class BaggingFromThePast_Cache_PI(Bandit_Algorithm_PI):
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

        # Cached Algo
        self.bootstrap_loss_estimates = np.zeros(shape = self.K)
        self.bootstrap_simulations_matrix = np.zeros(shape = (self.K, self.num_bootstrap_simulations))
        self.a_t_minus_1 = -1

        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "Cached BP ({} bag size {})".format("adding" if add_bag else "subtracting", 
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

    def get_arm_to_pull(self, importance_weighted_losses, losses, t, historyIsRewards = False):
        if (t < self.exploration_phase_length):
            A_t = math.floor(t / self.data_generating_mechanism.get_init_exploration())
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
        else:
            num_counts = np.zeros(shape = self.K)
            
            for n in range(self.num_bootstrap_simulations):
                uniform_sample = list(random.choice(t, size = self.get_bag_size(t), replace=True))
                if (self.a_t_minus_1 == -1):
                    cumulative_losses = np.sum(importance_weighted_losses, axis = 1)
                    bag_losses = np.sum(importance_weighted_losses[:, uniform_sample], axis=1)
                    self.bootstrap_simulations_matrix[:, n] = (cumulative_losses + bag_losses) if self.add_bag else (cumulative_losses - bag_losses)
                else:
                    cumulative_losses = np.sum(importance_weighted_losses[self.a_t_minus_1, :])
                    bag_losses = np.sum(importance_weighted_losses[self.a_t_minus_1, uniform_sample])
                    self.bootstrap_simulations_matrix[self.a_t_minus_1, n] = (cumulative_losses + bag_losses) if self.add_bag else (cumulative_losses - bag_losses)

                arm_chosen_this_simulation = np.argmax(self.bootstrap_simulations_matrix[:, n]) if historyIsRewards else np.argmin(self.bootstrap_simulations_matrix[:, n])
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1
                if (n == self.num_bootstrap_simulations - 1):
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