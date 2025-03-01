import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI, Bandit_Algorithm_FI
import math

class BaggingFromThePast_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()

        self.__data_generating_mechanism = data_generating_mechanism
        self.__label = "Bootstrapping from Columns"

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
            uniform_sample = list(random.randint(low = 0, high = t, size = self.T - t))
            arm_estimates_current_round = np.mean(losses[:, list(range(t)) + uniform_sample], axis = 1)
            return np.argmin(arm_estimates_current_round)
        
class BaggingFromThePast_PI(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.num_bootstrap_simulations = 100

        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "Bootstrapping from Columns"

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
                uniform_sample = list(random.randint(low = 0, high = t, size = self.T - t))
                arm_estimates_current_simulation = np.mean(importance_weighted_losses[:, list(range(t)) + uniform_sample], axis = 1)
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
        
class BaggingFromThePast_FI_v0(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.num_bootstrap_simulations = 100
        self.__label = "Bootstrapping from Individual Losses"

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
            arm_estimates_current_round = np.zeros(shape = self.K)

            for j in range(self.K):
                uniform_sample = list(random.randint(low = 0, high = t, size = self.T - t))
                arm_estimates_current_round[j] = np.mean(losses[j, list(range(t)) + uniform_sample])
        
            return np.argmin(arm_estimates_current_round)

# v_0 involves just sampling individual loss co-ordinates instead 
# of column vectors and also does not have any exploration component
class BaggingFromThePast_PI_v0(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.num_bootstrap_simulations = 100

        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "Bootstrapping from Individual Losses"

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
                arm_estimates_current_simulation = np.zeros(shape = self.K)

                for j in range(self.K):
                    uniform_sample = list(random.randint(low = 0, high = t, size = self.T - t))
                    arm_estimates_current_simulation[j] = np.mean(importance_weighted_losses[j, list(range(t)) + uniform_sample])

                arm_chosen_this_simulation = np.argmin(arm_estimates_current_simulation)
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1

            if (self.num_bootstrap_simulations == n + 1):
                A_t = arm_chosen_this_simulation

            self.current_sampling_distribution = num_counts / sum(num_counts)
            
        return A_t

# v_1 does not have the exploration component
class BaggingFromThePast_PI_v1(Bandit_Algorithm_PI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.num_bootstrap_simulations = 100

        self.__data_generating_mechanism = data_generating_mechanism
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__label = "Bootstrapping from Columns w/o exploration"

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
                uniform_sample = list(random.randint(low = 0, high = t, size = self.T - t))
                arm_estimates_current_simulation = np.mean(importance_weighted_losses[:, list(range(t)) + uniform_sample], axis = 1)
                arm_chosen_this_simulation = np.argmin(arm_estimates_current_simulation)
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1
                
                if (self.num_bootstrap_simulations == n + 1):
                    A_t = arm_chosen_this_simulation
                
            self.current_sampling_distribution = num_counts / sum(num_counts)
        
        return A_t

