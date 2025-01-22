import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math


def GIRO_arm_estimates_factory_function(A):

    def GIRO_arm_estimates_function(history, t, N):
        num_arms = len(history)
        arm_estimates_current_round = np.zeros(shape=K)
        for j in range(num_arms):

            s = len(history[j])
            if (s > 0): 
                # V_is stores the number 
                # of observed-rewards 
                # and pseudo-rewards
                V_is = np.count_nonzero(history[j])
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

        return arm_estimates_current_round

    return GIRO_arm_estimates_function


def UCB_arm_estimates_function(history, t, N):

    num_arms = len(history)
    arm_estimates_current_round = np.zeros(shape=K)
    for j in range(num_arms):

        s = len(history[j])

        if (s > 0):
            f_hat_pi = np.mean(history[j])
            f_t = 1 + (t * (math.log(t)**2))
            half_confidence_interval_width = math.sqrt((2 * math.log(f_t)) / (s))
            arm_estimates_current_round[j] = f_hat_pi + half_confidence_interval_width
        else:
            arm_estimates_current_round[j] = np.inf

    return arm_estimates_current_round


def TS_arm_estimates_function(history, t, N):
    num_arms = len(history)
    arm_estimates_current_round = np.zeros(shape=K)

    for j in range(num_arms):
        s = len(history[j])
        alpha_j = np.count_nonzero(history[j])
        beta_j = s - alpha_j
        arm_estimates_current_round[j] = random.beta(alpha_j + 1, beta_j + 1, size=1)

    return arm_estimates_current_round

def bagging_from_future_arm_estimates(history, t, N):
    num_arms = len(history)
    arm_estimates_current_round = np.zeros(shape=K)

    for j in range(num_arms):
        s = len(history[j])

        if (s < 5):
            arm_estimates_current_round[j] = np.inf
        else:
            rewards = np.zeros(shape=N)
            uniform_sample = random.randint(low = 0, high = s-1, size = N - s)
            for i in range(N):
                if (i < s):
                    rewards[i] = history[j][i]
                else:
                    rewards[i] = history[j][uniform_sample[i-s]]
            arm_estimates_current_round[j] = np.mean(rewards)

    return arm_estimates_current_round

def bernoulli_bandit(mu_i):
    return random.binomial(n=1, p = mu_i)

def beta_bandit_factory(nu, ts=False):
    
    def beta_bandit(mu_i):
        r_t = random.beta(nu * mu_i, nu * (1 - mu_i))
        if (ts == True):
            return random.binomial(n=1, p = r_t)
        else:
            return r_t
    
    return beta_bandit

# K-armed Bernoulli Bandit 
# problem tackled using 
# various algorithms
def simulate_stochastic_bandit_problem(K, T, generate_reward, compute_arm_estimates, M=1000):

    # data_structure to keep track of the
    # accumulated regret up to the current round
    # for M different runs
    accumulated_regret = np.zeros(shape=(M, T))

    # Generate the means of the arms 
    # using a Uniform distribution
    arms_mean_lower_range = 0.25
    arms_mean_upper_range = 0.75
    arms_mean = np.flip(np.sort(random.uniform(arms_mean_lower_range, arms_mean_upper_range, size = K)))
    optimal_arm_mean = arms_mean[0]

    for m in range(0, M):
        # initialize the data_structures
        # to hold the history of each arm
        history = []
        for i in range(K):
            history.append([])

        for i in range(T):
            # estimate arm values 
            arm_estimates_current_round = compute_arm_estimates(history, i, T)

            # get maximum estimate 
            # of the pulled arm
            I_t = np.argmax(arm_estimates_current_round)
    
            # reward is generated 
            # by nature
            r_t = generate_reward(arms_mean[I_t])

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

        print('m = {:d}'.format(m))

    return np.mean(accumulated_regret, axis=0)


T = 500
K = 10

plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = bernoulli_bandit, compute_arm_estimates = bagging_from_future_arm_estimates), label="Average Regret (BF)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K= K, T = T, generate_reward = bernoulli_bandit, compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K= K, T = T, generate_reward = bernoulli_bandit, compute_arm_estimates = TS_arm_estimates_function), label="Average Regret (Thompson Sampling)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(0.5)), label="Average Regret (GIRO; A = 0.5)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(0.1)), label="Average Regret (GIRO; A = 0.1)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(1)), label="Average Regret (GIRO; A = 1)")
plt.title("Observed Regret of GIRO (with varying A) vs. UCB vs. Thompson Sampling vs. Bagging from Future on 10-armed Bernoulli Bandit problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=4), compute_arm_estimates = bagging_from_future_arm_estimates), label="Average Regret (BF)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K= K, T = T, generate_reward = beta_bandit_factory(nu=4), compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K= K, T = T, generate_reward = beta_bandit_factory(nu=4, ts=True), compute_arm_estimates = TS_arm_estimates_function), label="Average Regret (Thompson Sampling)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=4), compute_arm_estimates = GIRO_arm_estimates_factory_function(0.5)), label="Average Regret (GIRO; A = 0.5)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=4), compute_arm_estimates = GIRO_arm_estimates_factory_function(0.1)), label="Average Regret (GIRO; A = 0.1)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=4), compute_arm_estimates = GIRO_arm_estimates_factory_function(1)), label="Average Regret (GIRO; A = 1)")
plt.title("Observed Regret of GIRO (with varying A) vs. UCB vs. Thompson Sampling vs. Bagging from Future on 10-armed (high-variance) Beta Bandit problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=16), compute_arm_estimates = bagging_from_future_arm_estimates), label="Average Regret (BF)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K= K, T = T, generate_reward = beta_bandit_factory(nu=16), compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K= K, T = T, generate_reward = beta_bandit_factory(nu=16, ts=True), compute_arm_estimates = TS_arm_estimates_function), label="Average Regret (Thompson Sampling)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=16), compute_arm_estimates = GIRO_arm_estimates_factory_function(0.5)), label="Average Regret (GIRO; A = 0.5)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=16), compute_arm_estimates = GIRO_arm_estimates_factory_function(0.1)), label="Average Regret (GIRO; A = 0.1)")
plt.plot(range(T), simulate_stochastic_bandit_problem(K = K, T = T, generate_reward = beta_bandit_factory(nu=16), compute_arm_estimates = GIRO_arm_estimates_factory_function(1)), label="Average Regret (GIRO; A = 1)")
plt.title("Observed Regret of GIRO (with varying A) vs. UCB vs. Thompson Sampling vs. Bagging from Future on 10-armed (low-variance) Beta Bandit problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()
        
