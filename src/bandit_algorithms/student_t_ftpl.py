import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

import numpy as np
from numpy import random
from bandit_algorithms.algorithm_class import Bandit_Algorithm_FI
import math

class Student_T_FTPL_FI(Bandit_Algorithm_FI):
    def __init__(self, data_generating_mechanism):
        super().__init__()
        self.K = data_generating_mechanism.get_K()
        self.T = data_generating_mechanism.get_T()
        self.exploration_phase_length = data_generating_mechanism.get_exploration_phase_length()
        self.init_exploration = data_generating_mechanism.get_init_exploration()
        self.__label = "Student_t_FTPL"
        self.__data_generating_mechanism = data_generating_mechanism


    @property
    def data_generating_mechanism(self):
        return self.__data_generating_mechanism
    
    @property
    def label(self):
        return self.__label

    def get_arm_to_pull(self, losses, t, extra_param):
        if (t < self.exploration_phase_length):
            return math.floor(t / self.init_exploration)
        else:
            arm_estimates_current_round = np.mean(losses, axis = 1)
            student_t_perturbation = np.zeros(shape = self.K)

            for i in range(self.K):
                student_t_perturbation[i] = arm_estimates_current_round[i] + (math.sqrt(t) * np.random.standard_t(
                                                                                               df = 1, 
                                                                                               size = 1))
            
            return np.argmin(arm_estimates_current_round + student_t_perturbation)