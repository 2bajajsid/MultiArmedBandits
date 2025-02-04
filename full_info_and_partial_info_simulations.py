import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from itertools import chain
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


def UCB_factory(partial_info = False):

    def UCB_arm_estimates_function(importance_weighted_history, unweighted_history, t, N, k):
        if (t < 50):
            if (partial_info == False):
                return [math.floor(t / 5)]
            else:
                one_hot = np.zeros(shape = k)
                one_hot[math.floor(t / 5)] = 1
                return [math.floor(t / 5), one_hot]

        arm_estimates_current_round = np.zeros(shape=K)

        for j in range(k):
            s = len(unweighted_history[j])

            if (s > 0):
                f_hat_pi = (1 - np.mean(unweighted_history[j]))
                f_t = 1 + (t * (math.log(t)**2))
                half_confidence_interval_width = math.sqrt((2 * math.log(f_t)) / (s))
                arm_estimates_current_round[j] = f_hat_pi + half_confidence_interval_width
            else:
                arm_estimates_current_round[j] = np.inf

        if (partial_info == False):
            return np.argmax(arm_estimates_current_round)
        else:
            one_hot = np.zeros(shape = k)
            one_hot[np.argmax(arm_estimates_current_round)] = 1
            return [np.argmax(arm_estimates_current_round), one_hot]
        
    return UCB_arm_estimates_function


def TS_arm_estimates_function(importance_weighted_history, unweighted_history, t, N, k):
    if (t < 50):
        one_hot = np.zeros(shape = k)
        one_hot[math.floor(t / 5)] = 1
        return [math.floor(t / 5), one_hot]
    
    arm_estimates_current_round = np.zeros(shape = k)

    for j in range(k):
        s = len(unweighted_history[j])
        alpha_j = np.sum(unweighted_history[j] == 0)
        beta_j = s - alpha_j
        arm_estimates_current_round[j] = random.beta(alpha_j + 1, beta_j + 1, size=1)

    one_hot = np.zeros(shape = k)
    one_hot[np.argmax(arm_estimates_current_round)] = 1
    return [np.argmax(arm_estimates_current_round), one_hot]

def bagging_from_past_into_future_factory(initial_exploration, partial_info = False):

    def bagging_from_past_into_future(importance_weighted_history, stochastic_history, t, N, k):
        if (t < initial_exploration * k):
            if (partial_info == False):
                return math.floor(t / initial_exploration)
            else:
                one_hot = np.zeros(shape = k)
                one_hot[math.floor(t / initial_exploration)] = 1
                return [math.floor(t / initial_exploration), one_hot]

        num_counts = np.zeros(shape = k)
        num_bootstrap_simulations = 100

        A_t = 0

        if (partial_info == False):
            uniform_sample = list(random.randint(low = 0, high = t, size = N - t))
            arm_estimates_current_round = np.mean(stochastic_history[:, list(range(t)) + uniform_sample], axis = 1)
            return np.argmin(arm_estimates_current_round)

        for n in range(num_bootstrap_simulations):
            uniform_sample = list(random.randint(low = 0, high = t, size = N - t))
            arm_estimates_current_simulation = np.mean(importance_weighted_history[:, list(range(t)) + uniform_sample], axis = 1)
            arm_chosen_this_simulation = np.argmax(-1 * arm_estimates_current_simulation)
            num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1
            if (num_bootstrap_simulations == n + 1):
                A_t = arm_chosen_this_simulation
                
        p_t = num_counts / num_bootstrap_simulations
        
        # flip a bernoulli with success 
        # probability of 1/t to add an 
        # explicit exploration component
        z = np.random.binomial(n = 1, p = (1 / t))
        p_t = ((1 - (1 / t)) * p_t) + (1 / (t * k))

        if (z == 1):
            return [random.randint(low = 0, high = k, size = 1), p_t]
        else:
            return [A_t, p_t]

    return bagging_from_past_into_future


def explore_then_commit_factory(initial_exploration, partial_info = False):

    def explore_then_commit(importance_weighted_history, unweighted_history, t, N, k):

        if (t < (initial_exploration * k)):
                one_hot = np.zeros(shape = k)
                one_hot[math.floor(t / initial_exploration)] = 1
                return [math.floor(t / initial_exploration), one_hot]
        
        arm_estimates_current_round = np.zeros(shape = k)
        for i in range(k):
            arm_estimates_current_round[i] = (1-np.mean(unweighted_history[i]))

        if (partial_info == False):
            return np.argmax(arm_estimates_current_round)
        else:
            one_hot = np.zeros(shape = k)
            one_hot[np.argmax(arm_estimates_current_round)] = 1
            return [np.argmax(arm_estimates_current_round), one_hot]
    
    return explore_then_commit

def hedge(importance_weighted_history, unweighted_history, t, N, k):
    neta = math.sqrt((2 * math.log(k)) / N)
    
    cumulative_losses = np.sum(unweighted_history, axis = 1)
    for i in range(k):
        cumulative_losses[i] = math.exp(-1 * neta * cumulative_losses[i])

    normalization_constant = np.sum(cumulative_losses)
    prob_distr = cumulative_losses / normalization_constant 

    return [np.random.choice(K, p = prob_distr)]

def exp3(importance_weighted_history, unweighted_history, t, N, k):
    neta = math.sqrt((2 * math.log(K)) / (N * K))

    cumulative_losses = np.sum(importance_weighted_history, axis = 1)
    for i in range(k):
        cumulative_losses[i] = math.exp(-1 * neta * cumulative_losses[i])

    normalization_constant = np.sum(cumulative_losses)
    prob_distr = cumulative_losses / normalization_constant

    a_t = np.random.choice(K, p = prob_distr)

    return [a_t, prob_distr]

# Zimmert & Seldin (2017)
# Improved Parametrization of Exp3++
def exp3_plus_plus(importance_weighted_history, unweighted_history, t, N, k):

    if (t < 50):
        one_hot = np.zeros(shape = k)
        one_hot[math.floor(t / 5)] = 1
        return [math.floor(t / 5), one_hot]
    
    alpha = 3
    beta = 256

    # data structures that mimic
    # the pseudo-code of Algorithm 2: Gap  
    # Estimation in Randomized Playing Strategies
    eta_t = np.zeros(shape = k)
    epsilon_t = np.zeros(shape = k)
    N_t_minus_one = np.zeros(shape = k)
    UCB_t = np.zeros(shape = k)
    LCB_t = np.zeros(shape = k)
    delta_t = np.zeros(shape = k)

    for a in range(k):
        N_t_minus_one[a] = len(unweighted_history[a])

    neta_t = 1/2 * (math.sqrt(math.log(k) / t * k))
    cumulative_losses = np.sum(importance_weighted_history, axis = 1)
    for a in range(k):
        cumulative_losses[a] = math.exp(-1 * neta_t * cumulative_losses[a])
        margin_confidence_interval = math.sqrt((alpha * math.log(t * (K)**(1 / alpha))) / 2 * N_t_minus_one[a])
        UCB_t[a] = min(1, (cumulative_losses[a] / N_t_minus_one[a]) + margin_confidence_interval)
        LCB_t[a] = max(0, (cumulative_losses[a] / N_t_minus_one[a]) - margin_confidence_interval)

    for a in range(k):
        delta_t[a] = max(0, LCB_t[a] - min(UCB_t))
        eta_t[a] = (beta * math.log(t)) / (t * (delta_t[a])**2)
        epsilon_t[a] = min(1 / (2*k), (1 / 2) * (math.sqrt(math.log(k) / (t * k))), eta_t[a])

    normalization_constant = np.sum(cumulative_losses)
    p_t = cumulative_losses / normalization_constant

    sum_epsilon_t = np.sum(epsilon_t)

    for a in range(k):
        p_t[a] = ((1 - sum_epsilon_t) * p_t[a]) + (epsilon_t[a])

    a_t = np.random.choice(K, p = p_t)
    return [a_t, p_t]

def tsallis_inf_factory(partial_info = False):

    def tsallis_inf(importance_weighted_history, unweighted_history, t, N, k):
        neta_t = 2 * math.sqrt(1 / (t + 1))

        if (t < 50):
            if (partial_info == False):
                return [math.floor(t / 5)]
            else:
                one_hot = np.zeros(shape = k)
                one_hot[math.floor(t/5)] = 1
                return [np.argmax(one_hot), one_hot]

        cumulative_losses = np.sum(importance_weighted_history, axis = 1)
        p_t = compute_p_t(cumulative_losses, neta_t)
        p_t = p_t / sum(p_t)
        a_t = np.random.choice(k, p = p_t, size = 1)

        if (partial_info == False):
            return a_t
        else:
            return [a_t, p_t]

    return tsallis_inf

# Implementing Newton-Raphson helper for solving the
# OMD optimization step as specified in 
# Zimmert and Seldin (2022)
def compute_p_t(cumulative_losses, neta_t):
    curr_x = 1/2
    temp_x = np.inf 
    w_t = np.zeros(shape = len(cumulative_losses))

    while (abs(curr_x - temp_x) > 0.0005):
        if (temp_x < np.inf):
            curr_x = temp_x
        
        for i in range(len(cumulative_losses)):
            w_t[i] = 4 * (1 / (neta_t * (cumulative_losses[i] - curr_x))**2)
        
        denom = (neta_t * sum(w_t ** (3/2)))

        if (denom < 0.1):
            denom = 0.1

        temp_x = curr_x - ((sum(w_t) - 1) / denom)

    return w_t

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

def stochastically_constrained_adversarial(mu, t):
    
    rewards = np.zeros(shape = len(mu))
    for i in range(len(mu)):
        rewards[i] = np.random.binomial(n = 1, p = mu[i])
    
    if (t < 100 or (t > 300 and t < 600)):
        return rewards
    else:
        temp = rewards[7]
        rewards[7] = rewards[4]
        rewards[4] = temp
        return rewards

# simulate K-armed
# full information problem 
# and return accumulated regret
def simulate_full_information_problem(K, T, generate_rewards, get_arm_to_pull, M=100):

    # data_structure to keep track of the
    # accumulated regret up to the current round
    # for M different runs
    accumulated_regret = np.zeros(shape=(M, T))

    for m in range(0, M):
        # initialize the data_structures
        # to hold the history of each arm
        history = np.zeros(shape = (K, T))
        current_loss = np.zeros(shape = T)

        for t in range(T):
            # estimate arm values 
            I_t = get_arm_to_pull(history, history, t, T, K)

            # reward is  
            # generated by
            # nature
            r_t = generate_rewards(arms_mean, t)

            # Update the statistics
            for i in range(K): 
                history[i][t] = 1 - r_t[i]
            
            current_loss[t] = 1 - r_t[I_t]
            
        cumulative_losses = np.sum(history, axis=1)
        optimal_arm = np.argmin(cumulative_losses)

        for t in range(T):
            if (t == 0):
                accumulated_regret[m][t] = current_loss[t] - history[optimal_arm][t]
            else: 
                accumulated_regret[m][t] = accumulated_regret[m][t-1] + (current_loss[t] - history[optimal_arm][t])

        print('m = {:d}'.format(m))

    return np.mean(accumulated_regret, axis=0)


# simulate K-armed
# partial information problem 
# and return accumulated regret
def simulate_partial_information_problem(K, T, generate_rewards, get_arm_and_weight_to_pull, M=100):

    # data_structure to keep track of the
    # accumulated regret up to the current round
    # for M different runs
    accumulated_regret = np.zeros(shape=(M, T))

    for m in range(0, M):
        # initialize the data structures
        # to hold the losses of each arm
        importance_weighted_history = np.zeros(shape = (K, T))
        unweighted_history = []
        full_history = np.zeros(shape = (K, T))
        P_ti = np.zeros(shape = (K, T))

        for i in range(K):
            unweighted_history.append([])

        for t in range(T):
            # estimate arm values 
            arm_and_weight = get_arm_and_weight_to_pull(importance_weighted_history, unweighted_history, t, T, K)
            I_t = arm_and_weight[0]
            P_ti[:, t] = arm_and_weight[1]

            # reward is  
            # generated by
            # nature
            r_t = generate_rewards(arms_mean, t)

            # update the statistics
            # only add non-zero value to the history
            # if arm was chosen by the learner
            # to simulate partial info
            for i in range(K): 
                if (i == I_t):
                    importance_weighted_history[i][t] = ((1 - r_t[i]) / P_ti[i][t])
                    unweighted_history[i].append(1 - r_t[i])
                else:
                    importance_weighted_history[i][t] = 0
                full_history[i][t] = 1 - r_t[i]
            
        cumulative_losses = np.sum(full_history, axis=1)
        optimal_arm = np.argmin(cumulative_losses)

        for t in range(T):
            if (t == 0):
                accumulated_regret[m][t] = (np.dot(P_ti[:, t], full_history[:, t])) - full_history[optimal_arm][t]
            else: 
                accumulated_regret[m][t] = accumulated_regret[m][t-1] + (np.dot(P_ti[:, t], full_history[:, t]) - full_history[optimal_arm][t])

        print('m = {:d}'.format(m))

    return np.mean(accumulated_regret, axis=0)


T = 1000
K = 10

arms_mean = np.random.uniform(low = 0.1, high = 0.5, size = K)
arms_mean[4] = 0.54
arms_mean[7] = 0.86

plt.rcParams["figure.figsize"] = (15,6)

plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), label="Bagging from the Past")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = tsallis_inf_factory(partial_info=True)), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = exp3_plus_plus), label="Exp3++")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, get_arm_and_weight_to_pull = UCB_factory(partial_info = True)), label="UCB")
plt.title("Averaged Regret on a Partial-Information Stochastically Constrained Adversarial Problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI Stochastically Constrained Adversarial Regime')
plt.close()

arms_mean = np.random.uniform(low = 0.1, high = 0.5, size = K)
arms_mean[4] = 0.75

plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = tsallis_inf_factory(partial_info = False)), label="Tsallis-Inf")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = bagging_from_past_into_future_factory(5, partial_info=False)), label="Bagging from the Past")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = hedge), label="Hedge")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_to_pull = UCB_factory(partial_info = False)), label="UCB")

plt.legend()
plt.title("Averaged Regret on the Full-Information Stochastic Bernoulli Bandit problem (Delta > 0.25)")
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('FI High-Gap Stochastic Problem')
plt.close() 

plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), label="Bagging from the Past")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = explore_then_commit_factory(5, partial_info=True)), label="ETC")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = UCB_factory(partial_info=True)), label="UCB")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = tsallis_inf_factory(partial_info=True)), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3_plus_plus), label="Exp3++")
plt.title("Averaged Regret on the Partial-Information Stochastic Bernoulli Bandit problem (Delta > 0.25)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI High-Gap Stochastic Problem')
plt.close() 