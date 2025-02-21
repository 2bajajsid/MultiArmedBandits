import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_PI
from scipy import optimize
import math

def obj(x, cumulative_losses, neta_t):
        return sum(get_w(x, cumulative_losses, neta_t)) - 1
    
def get_w(x, cumulative_losses, neta_t):
    min_loss = min(cumulative_losses)
    w = np.zeros(shape = len(cumulative_losses))
    for i in range(len(cumulative_losses)):
        w[i] = 4 / ((neta_t * (cumulative_losses[i] - min_loss) + x)**2)
    return w

class Tsallis_Inf(Bandit_Algorithm_PI):

    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.__current_sampling_distribution = np.ones(shape = self.K) / self.K
        self.__data_generating_mechanism = data_generating_mechanism
        self.__label = "Tsallis_Inf"
        self.x = 0

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
        neta_t = 2 * math.sqrt(1 / (t + 1))
        if (t < self.data_generating_mechanism.get_exploration_phase_length()):
            A_t = math.floor(t / self.data_generating_mechanism.get_init_exploration())
            self.current_sampling_distribution = np.zeros(shape = self.data_generating_mechanism.get_K())
            self.current_sampling_distribution[A_t] = 1
        else:
            cumulative_losses = np.sum(importance_weighted_losses, axis = 1)
            x = optimize.bisect(obj, 1, 
                                2 * math.sqrt(self.data_generating_mechanism.get_K()), 
                                args = (cumulative_losses, neta_t))

            self.current_sampling_distribution = get_w(x, cumulative_losses, neta_t)
            self.current_sampling_distribution = self.current_sampling_distribution / sum(self.current_sampling_distribution)
            A_t = np.random.choice(self.K, p = self.current_sampling_distribution, size = 1)

        return A_t

