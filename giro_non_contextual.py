import numpy as np
import matplotlib.pyplot as plt
from numpy import random
import math

random.seed(895)

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

        return np.argmax(arm_estimates_current_round)

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

    return np.argmax(arm_estimates_current_round)


def TS_arm_estimates_function(history, t, N):
    num_arms = len(history)
    arm_estimates_current_round = np.zeros(shape=K)

    for j in range(num_arms):
        s = len(history[j])
        alpha_j = np.count_nonzero(history[j])
        beta_j = s - alpha_j
        arm_estimates_current_round[j] = random.beta(alpha_j + 1, beta_j + 1, size=1)

    return np.argmax(arm_estimates_current_round)

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

        return np.argmax(arm_estimates_current_round)
    
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

        return np.argmax(arm_estimates_current_round)
    
    return explore_then_commit

def hedge(history, t, N):
    k = np.shape(history)[0]
    neta = math.sqrt((2 * math.log(k)) / T)
    
    cumulative_losses = np.sum(history, axis = 1)
    for i in range(k):
        cumulative_losses[i] = math.exp(-1 * neta * (t - cumulative_losses[i]))

    normalization_constant = np.sum(cumulative_losses)
    prob_distr = cumulative_losses / normalization_constant 

    return np.random.choice(K, p = prob_distr)

def bernoulli_bandit(mu, t):
    k = len(mu)
    rewards = np.zeros(shape = k)

    for i in range(k):
        rewards[i] = random.binomial(n = 1, p = mu[i])

    return rewards

def beta_bandit_factory(nu, ts=False):
    
    def beta_bandit(mu, t):
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

def adversarial_data_1(mu, t):
    k = len(mu)
    rewards = np.zeros(shape = k)
    rewards[t % k] = 1
    return rewards

def adversarial_data_2(mu, t):
    k = len(mu)
    rewards = np.zeros(shape = k)
    rewards[math.floor(math.fmod(t / 10, k))] = 1
    return rewards

def adversarial_data_3(mu, t):
    rewards = np.random.uniform(low = 0, high = 1, size = len(mu))
    return rewards

# simulate K-armed stochastic 
# full information problem 
# and return accumulated regret
def simulate_full_information_problem(K, T, generate_rewards, get_arm_to_pull, M=100, ts = False):

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
                I_t = get_arm_to_pull(ts_history, t, T)
            else:
                I_t = get_arm_to_pull(history, t, T)
    
            # reward is  
            # generated by
            # nature
            r_t = generate_rewards(arms_mean, t)
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


T = 1000
K = 10

""" # Let the means of all the arms be 0.5 to simulate adversarial data
arms_mean = np.full(shape = K, fill_value = 0.5)

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = adversarial_data_1, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = adversarial_data_1, get_arm_to_pull = bagging_from_past_into_future_factory(5)), label="Bagging from the Past")
plt.title("Averaged Regret on the Full-Information Adversarial Bandit problem 1")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = adversarial_data_2, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = adversarial_data_2, get_arm_to_pull = bagging_from_past_into_future_factory(5)), label="Bagging from the Past")
plt.title("Averaged Regret on the Full-Information Adversarial Bandit problem 2")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = adversarial_data_3, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = adversarial_data_3, get_arm_to_pull = bagging_from_past_into_future_factory(5)), label="Bagging from the Past")
plt.title("Averaged Regret on the Full-Information Adversarial Bandit problem 3")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show() 

arms_mean = np.full(shape = K, fill_value = 0.5)

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = bagging_from_past_into_future_factory(5)), label="Bagging from the Past")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = explore_then_commit_factory(5)), label="Explore-then-Commit")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = UCB_arm_estimates_function), label="Upper-Confidence-Bound")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = TS_arm_estimates_function), label="Thompson-Sampling")
plt.title("Averaged Regret on the Full-Information Stochastic Bernoulli Bandit problem (No Gap)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()

arms_mean[4] = 0.52

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = bagging_from_past_into_future_factory(5)), label="Bagging from the Past")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = explore_then_commit_factory(5)), label="Explore-then-Commit")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = UCB_arm_estimates_function), label="Upper-Confidence-Bound")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = TS_arm_estimates_function), label="Thompson-Sampling")
plt.title("Averaged Regret on the Full-Information Stochastic Bernoulli Bandit problem (Low Gap)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show() """

arms_mean = np.random.uniform(low = 0.1, high = 0.5, size = K)
arms_mean[4] = 0.75

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = bagging_from_past_into_future_factory(5)), label="Bagging from the Past")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = explore_then_commit_factory(5)), label="Explore-then-Commit")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = UCB_arm_estimates_function), label="Upper-Confidence-Bound")
plt.plot(range(T), simulate_full_information_problem(K= K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = TS_arm_estimates_function), label="Thompson-Sampling")
plt.title("Averaged Regret on the Full-Information Stochastic Bernoulli Bandit problem (High Gap)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.show()