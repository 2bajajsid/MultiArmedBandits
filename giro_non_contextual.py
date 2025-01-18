import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

# expected regret upper bound of GIRO
# for Bernoulli Bandits
def get_expected_regret_upper_bound(time_horizon, a, k, means):
    b = (2*a + 1) / (a * (a + 1))
    c = ((2 * math.exp(2) * math.sqrt((2 * a) + 1)) / math.sqrt(2 * math.pi)) * math.exp((8 * b) / (2 - b)) * (1 + math.sqrt((2 * math.pi) / (4 - (2 * b)))) 

    # R(n) <= x + (ln(n) * y) for a > 0.707
    x = 0
    y = 0
    optimality_gaps = np.zeros(shape = k - 1)

    # Let's compute x and y
    for i in range(k-1):
        optimality_gaps[i] = means[0] - means[i+1]
        x += (4 * optimality_gaps[i])
        y += ((48 * a) + 8 + (16 * c)) / optimality_gaps[i]

    expected_regret_upper_bound = np.zeros(shape = time_horizon)
    for i in range(time_horizon):
        if (i == 0):
            expected_regret_upper_bound[i] = x
        else:
            expected_regret_upper_bound[i] = expected_regret_upper_bound[i-1] + (math.log(i+1) - math.log(i)) * (y)

    return expected_regret_upper_bound

# K-armed Bernoulli Bandit 
# problem tackled using GIRO 
# where 'A' negative and positive 
# pseudo rewards are added at each 
# step over T rounds
def conduct_giro_simulation(K, A, T, M=100):

    # data_structure to keep track of the
    # accumulated regret up to the current round
    # for M different runs
    accumulated_regret = np.zeros(shape=(M, T))

    for m in range(0, M):
        # Generate the means of the arms 
        # using a Uniform distribution
        arms_mean_lower_range = 0.25
        arms_mean_upper_range = 0.75
        arms_mean = np.flip(np.sort(random.uniform(arms_mean_lower_range, arms_mean_upper_range, size = K)))
        optimal_arm_mean = arms_mean[0]

        # initialize the data_structures
        # to hold the history of each arm
        history = []
        for i in range(K):
            history.append([])

        for i in range(T):
            # estimate arm values 
            arm_estimates_current_round = np.zeros(shape = K)

            for j in range(K):
                s = len(history[j])
                if (s > 0): 
                    # V_is stores the number 
                    # of observed rewards 
                    # and pseudo rewards
                    V_is = np.count_nonzero(history[j] == 1)
                    if int(A) == A:        
                        V_is += (A * s)
                        alpha = (2*A + 1)
                    else:
                        z = random.binomial(1, math.ceil(A*s) - (A*s))
                        if (z == 1):
                            V_is += (math.floor(A*s))
                            alpha = (2 * (math.floor(A*s)) + 1)
                        else:
                            V_is += (math.ceil(A*s))
                            alpha = (2 * (math.ceil(A*s)) + 1)
                    # U_is is the bootstrap sample
                    U_is = random.binomial(alpha * s, V_is / (alpha * s))
                    arm_estimates_current_round[j] = (U_is / (alpha * s))
                else:
                    arm_estimates_current_round[j] = np.inf

            # get maximum estimate 
            # of the pulled arm
            I_t = np.argmax(arm_estimates_current_round)
    
            # reward is generated 
            # by nature
            r_t = random.binomial(1, arms_mean[I_t])

            # Update the statistics 
            history[I_t].append(r_t)

            # add to accumulated regret only if 
            # optimal arm was not chosen this round 
            if (I_t != 0):
                sub_optimality_gap = (optimal_arm_mean - arms_mean[I_t]) 
                if (i == 0):
                    accumulated_regret[m][i] = sub_optimality_gap
                else: 
                    accumulated_regret[m][i] = accumulated_regret[m][i-1] + (sub_optimality_gap)
            else:
                if (i != 1):
                    accumulated_regret[m][i] = accumulated_regret[m][i-1]

        print('A={:f}, m = {:d}'.format(A, m))

    return np.mean(accumulated_regret, axis=0)

T = 5000
plt.plot(range(T), conduct_giro_simulation(K = 10, A = 0.5, T = T), label="Observed Regret with A = 0.5")
plt.plot(range(T), conduct_giro_simulation(K = 10, A = 1, T = T), label="Observed Regret with A = 1")
plt.plot(range(T), conduct_giro_simulation(K = 10, A = 2, T = T), label="Observed Regret with A = 2")
plt.plot(range(T), conduct_giro_simulation(K = 10, A = 5, T = T), label="Observed Regret with A = 5")
plt.plot(range(T), conduct_giro_simulation(K = 10, A = 10, T = T), label="Observed Regret with A = 10")
plt.plot(range(T), conduct_giro_simulation(K = 10, A = 10, T = T), label="Observed Regret with A = 50")
#plt.plot(range(T), get_expected_regret_upper_bound(T, A, K, arms_mean), label="Regret Upper Bound")
plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.title("Observed Regret of non-contextual GIRO on 10-armed Bernoulli Bandit problem with varying A")
plt.show()
        
