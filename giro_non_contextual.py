import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

random.seed(885)

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

def bagging_from_past_into_future_factory(initial_exploration):

    def bagging_from_past_into_future(history, t, N):
        num_arms = len(history)
        arm_estimates_current_round = np.zeros(shape=K)

        for j in range(num_arms):
            s = len(history[j])

            if (s < initial_exploration):
                arm_estimates_current_round[j] = np.inf
            else:
                rewards = np.zeros(shape=N)
                uniform_sample = random.randint(low = 0, high = s, size = N - s)
                for i in range(N):
                    if (i < s):
                        rewards[i] = history[j][i]
                    else:
                        rewards[i] = history[j][uniform_sample[i-s]]
                arm_estimates_current_round[j] = np.mean(rewards)

        return arm_estimates_current_round
    
    return bagging_from_past_into_future

def explore_then_commit_factory(initial_exploration):

    def explore_then_commit(history, t, N):
        num_arms = len(history)
        arm_estimates_current_round = np.zeros(shape=K)

        for j in range(num_arms):
            s = len(history[j])

            if (s < initial_exploration):
                arm_estimates_current_round[j] = np.inf
            else:
                arm_estimates_current_round[j] = np.mean(history[j])
        return arm_estimates_current_round
    
    return explore_then_commit


def bernoulli_bandit(mu):
    k = len(mu)
    rewards = np.zeros(shape = k)

    for i in range(k):
        rewards[i] = random.binomial(n = 1, p = mu[i])

    return rewards

def beta_bandit_factory(nu, ts=False):
    
    def beta_bandit(mu):
        k = len(mu)
        rewards = np.zeros(shape = k)

        for i in range(k):
            rewards[i] = random.beta(nu * mu[i], nu * (1 - mu[i]))

        if (ts == False):
            return rewards
        else:
            zero_one_rewards = np.zeros(shape = k)
            for i in range(K):
                zero_one_rewards[i] = random.binomial(n = 1, p = rewards[i])
            return [rewards, zero_one_rewards]
    
    return beta_bandit

# simulate K-armed stochastic 
# full information problem 
# and return accumulated regret
def simulate_full_information_problem(K, T, generate_rewards, compute_arm_estimates, M=100, ts = False):

    # data_structure to keep track of the
    # accumulated regret up to the current round
    # for M different runs
    accumulated_regret = np.zeros(shape=(M, T))

    for m in range(0, M):
        # initialize the data_structures
        # to hold the history of each arm
        history = []
        for i in range(K):
            history.append([])

        ts_history = []
        for i in range(K):
            ts_history.append([])

        current_reward = np.zeros(shape = T)

        for t in range(T):
            # estimate arm values 
            if (ts == True):
                arm_estimates_current_round = compute_arm_estimates(ts_history, t, T)
            else:
                arm_estimates_current_round = compute_arm_estimates(history, t, T)

            # get maximum estimate 
            # of the pulled arm
            I_t = np.argmax(arm_estimates_current_round)
    
            # reward is generated 
            # by nature
            r_t = generate_rewards(arms_mean)
            if (ts == True):
                z_t = r_t[1]
                r_t = r_t[0]

            # Update the statistics
            for i in range(K): 
                history[i].append(r_t[i])

                if (ts == True):
                    ts_history[i].append(z_t[i])
            
            current_reward[t] = r_t[I_t]
            
        cumulative_rewards = np.sum(history, axis=1)
        optimal_arm = np.argmax(cumulative_rewards)

        for t in range(T):
            if (t == 0):
                accumulated_regret[m][t] = history[optimal_arm][t] - current_reward[t] 
            else: 
                accumulated_regret[m][t] = accumulated_regret[m][t-1] + (history[optimal_arm][t] - current_reward[t])

        print('m = {:d}'.format(m))

    return np.mean(accumulated_regret, axis=0)


T = 500
K = 10

# Let the means of all the arms be 0.5 to simulate adversarial data
""" arms_mean = np.full(shape = K, fill_value = 0.333)

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = bagging_from_past_into_future_factory(5)), label="Average Regret (BF)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = explore_then_commit_factory(5)), label="Average Regret (ETC)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = TS_arm_estimates_function), label="Average Regret (Thompson Sampling)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(1)), label="Average Regret (GIRO; A = 1)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(0.1)), label="Average Regret (GIRO; A = 0.1)")
plt.title("Averaged Regret on the Full-Information Adversarial Bernoulli Bandit problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=4), compute_arm_estimates = bagging_from_past_into_future_factory(5)), label="Average Regret (BF)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=4), compute_arm_estimates = explore_then_commit_factory(5)), label="Average Regret (ETC)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=4), compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=4, ts=True), compute_arm_estimates = TS_arm_estimates_function, ts=True), label="Average Regret (Thompson Sampling)")
plt.title("Averaged Regret on the Full-Information Adversarial Beta Bandit (high-variance) problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=16), compute_arm_estimates = bagging_from_past_into_future_factory(5)), label="Average Regret (BF)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=16), compute_arm_estimates = explore_then_commit_factory(5)), label="Average Regret (ETC)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=16), compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=16, ts=True), compute_arm_estimates = TS_arm_estimates_function, ts=True), label="Average Regret (Thompson Sampling)")
plt.title("Averaged Regret on the Full-Information Adversarial Beta Bandit (low-variance) problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show() """

# Generate the means of the arms 
# using a Uniform distribution
arms_mean_lower_range = 0.25
arms_mean_upper_range = 0.75
arms_mean = random.uniform(arms_mean_lower_range, arms_mean_upper_range, size = K)
optimal_arm_mean_index = np.argmax(arms_mean)
optimal_arm_mean = arms_mean[optimal_arm_mean_index]

""" plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = bagging_from_past_into_future_factory(5)), label="Average Regret (BF)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = explore_then_commit_factory(5)), label="Average Regret (ETC)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = TS_arm_estimates_function), label="Average Regret (Thompson Sampling)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(1)), label="Average Regret (GIRO; A = 1)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, compute_arm_estimates = GIRO_arm_estimates_factory_function(0.1)), label="Average Regret (GIRO; A = 0.1)")
plt.title("Averaged Regret on the Full-Information Stochastic Bernoulli Bandit problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show() """

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=4), compute_arm_estimates = bagging_from_past_into_future_factory(5)), label="Average Regret (BF)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu = 4), compute_arm_estimates = explore_then_commit_factory(5)), label="Average Regret (ETC)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=4), compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=4, ts=True), compute_arm_estimates = TS_arm_estimates_function, ts=True), label="Average Regret (Thompson Sampling)")
plt.title("Averaged Regret on the Full-Information Stochastic Beta Bandit (high-variance) problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=16), compute_arm_estimates = bagging_from_past_into_future_factory(5)), label="Average Regret (BF)")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = beta_bandit_factory(nu=16), compute_arm_estimates = explore_then_commit_factory(5)), label="Average Regret (ETC)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=16), compute_arm_estimates = UCB_arm_estimates_function), label="Average Regret (UCB)")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = beta_bandit_factory(nu=16, ts=True), compute_arm_estimates = TS_arm_estimates_function, ts=True), label="Average Regret (Thompson Sampling)")
plt.title("Averaged Regret on the Full-Information Stochastic Beta Bandit (low-variance) problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()