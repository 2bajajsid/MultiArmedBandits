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


def UCB_factory(partial_info = False):

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

        if (partial_info == False):
            return np.argmax(arm_estimates_current_round)
        else:
            return [np.argmax(arm_estimates_current_round), 1]
        
    return UCB_arm_estimates_function


def TS_arm_estimates_function(history, t, N):
    num_arms = len(history)
    arm_estimates_current_round = np.zeros(shape=K)

    for j in range(num_arms):
        s = len(history[j])
        alpha_j = np.count_nonzero(history[j])
        beta_j = s - alpha_j
        arm_estimates_current_round[j] = random.beta(alpha_j + 1, beta_j + 1, size=1)

    return np.argmax(arm_estimates_current_round)

def bagging_from_past_into_future_factory(initial_exploration, partial_info = False):

    if (partial_info == False):

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
    
    else:

        def bagging_from_past_into_future_partial(history, t, N):
            num_arms = len(history)

            if (t < 50):
                return [math.floor(t / 5), 1]

            num_counts = np.zeros(shape = num_arms)
            num_bootstrap_simulations = 30

            for n in range(num_bootstrap_simulations):

                arm_estimates_current_round = np.zeros(shape=K)
                for j in range(num_arms):
                    s = len(history[j])
                    losses = np.zeros(shape=N)
                    uniform_sample = random.randint(low = 0, high = s, size = N - s)
                    for i in range(N):
                        if (i < s):
                            losses[i] = history[j][i]
                        else:
                            losses[i] = history[j][uniform_sample[i-s]]
                
                    arm_estimates_current_round[j] = np.mean(losses)

                arm_chosen_this_simulation = np.argmax(-1 * arm_estimates_current_round)
                num_counts[arm_chosen_this_simulation] = num_counts[arm_chosen_this_simulation] + 1

            num_counts = num_counts / num_bootstrap_simulations

            A_t = np.random.choice(K, p = num_counts)
            return [A_t, num_counts[A_t]]

        return bagging_from_past_into_future_partial


def explore_then_commit_factory(initial_exploration, partial_info = False):

    def explore_then_commit(history, t, N):
        num_arms = len(history)
        arm_estimates_current_round = np.zeros(shape=K)

        for j in range(num_arms):
            s = len(history[j])

            if (s < initial_exploration):
                arm_estimates_current_round[j] = np.inf
            else:
                arm_estimates_current_round[j] = np.mean(history[j])

        if (partial_info == False):
            return np.argmax(arm_estimates_current_round)
        else:
            return [np.argmax(arm_estimates_current_round), 1]
    
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

def exp3(history, t, N):
    k = np.shape(history)[0]
    neta = math.sqrt((2 * math.log(K)) / (N * K))

    cumulative_losses = np.sum(history, axis = 1)
    for i in range(k):
        cumulative_losses[i] = math.exp(-1 * neta * cumulative_losses[i])

    normalization_constant = np.sum(cumulative_losses)
    prob_distr = cumulative_losses / normalization_constant

    a_t = np.random.choice(K, p = prob_distr)

    return [a_t, prob_distr[a_t]]

def exp3_ix(history, t, N):
    k = np.shape(history)[0]
    neta = math.sqrt((2 * math.log(K + 1)) / (N * K))
    gamma = neta / 2

    cumulative_losses = np.sum(history, axis = 1)
    for i in range(k):
        cumulative_losses[i] = math.exp(-1 * neta * cumulative_losses[i])

    normalization_constant = np.sum(cumulative_losses)
    prob_distr = cumulative_losses / normalization_constant
    a_t = np.random.choice(K, p = prob_distr)

    return [a_t, prob_distr[a_t] + gamma]

def tsallis_inf(history, t, N):
    neta_t = 2 * math.sqrt(1 / (t + 1))

    if (t < 50):
        return [math.floor(t / 5), 1]

    cumulative_losses = np.sum(history, axis = 1)
    p_t = compute_p_t(cumulative_losses, neta_t)
    a_t = np.random.choice(K, p = p_t / sum(p_t))

    return [a_t, p_t[a_t]]

def compute_p_t(cumulative_losses, neta_t):
    curr_x = 1/2
    temp_x = np.inf 
    w_t = np.zeros(shape = len(cumulative_losses))

    while (abs(curr_x - temp_x) > 0.005):
        if (temp_x < np.inf):
            curr_x = temp_x
        
        for i in range(len(cumulative_losses)):
            w_t[i] = 4 * (1 / (neta_t * (cumulative_losses[i] - curr_x))**2)
        
        temp_x = curr_x - ((sum(w_t) - 1) / (neta_t * sum(w_t ** (3/2))))

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
def simulate_full_information_problem(K, T, generate_rewards, get_arm_to_pull, M=50, ts = False):

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


# simulate K-armed
# partial information problem 
# and return accumulated regret
def simulate_partial_information_problem(K, T, generate_rewards, get_arm_and_weight_to_pull, M=25, isStochastic = False):

    # data_structure to keep track of the
    # accumulated regret up to the current round
    # for M different runs
    accumulated_regret = np.zeros(shape=(M, T))

    for m in range(0, M):
        # initialize the data structures
        # to hold the history of each arm
        importance_weighted_history = []
        history = []
        stochastic_history = []
        for i in range(K):
            importance_weighted_history.append([])
            stochastic_history.append([])
            history.append([])

        current_losses = np.zeros(shape = T)

        for t in range(T):
            # estimate arm values 
            if (isStochastic == False):
                arm_and_weight = get_arm_and_weight_to_pull(importance_weighted_history, t, T)
            else:
                arm_and_weight = get_arm_and_weight_to_pull(stochastic_history, t, T)
            I_t = arm_and_weight[0]
            P_ti = arm_and_weight[1]
    
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
                    importance_weighted_history[i].append(((1 - r_t[i]) / P_ti))
                    stochastic_history[i].append(r_t[i])
                else:
                    importance_weighted_history[i].append(0)
                history[i].append(1 - r_t[i])
            
            current_losses[t] = (1 - r_t[I_t])
            
        cumulative_losses = np.sum(history, axis=1)
        optimal_arm = np.argmax(-1 * cumulative_losses)

        for t in range(T):
            if (t == 0):
                accumulated_regret[m][t] = (current_losses[t] - history[optimal_arm][t])
            else: 
                accumulated_regret[m][t] = accumulated_regret[m][t-1] + (current_losses[t] - history[optimal_arm][t])

        print('m = {:d}'.format(m))

    return np.mean(accumulated_regret, axis=0)


T = 1000
K = 10

arms_mean = np.random.uniform(low = 0.1, high = 0.5, size = K)
arms_mean[4] = 0.54
arms_mean[7] = 0.86

plt.rcParams["figure.figsize"] = (15,6)

""" plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), label="Bagging from the Past")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = stochastically_constrained_adversarial, 
                                                        get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.title("Averaged Regret on a Partial-Information Stochastically Constrained Adversarial Problem")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI SCA 1')
plt.close()

# Let the means of all the arms be 0.5 to simulate adversarial data
arms_mean = np.full(shape = K, fill_value = 0.5)

plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_1, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), label="Bagging from the Past")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_1, get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_1, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_1, get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.title("Averaged Regret on a Partial-Information Adversarial Bandit problem (cyclical 1)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI A1')
plt.close() """

""" plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_2, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), label="Bagging from the Past")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_2, get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_2, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_2, get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.title("Averaged Regret on a Partial-Information Adversarial Bandit problem (cyclical 2)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI A2')
plt.close() """

""" plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_3, get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), label="Bagging from the Past")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_3, get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_3, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = adversarial_data_3, get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.title("Averaged Regret on a Partial Information Adversarial Bandit problem (Random Uniform Data)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI A3')
plt.close() """

arms_mean = np.full(shape = K, fill_value = 0.5)

plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = UCB_factory(partial_info = True), isStochastic = True), label="Upper-Confidence-Bound")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = explore_then_commit_factory(5, partial_info = True), isStochastic = True), label="Explore-then-Commit")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True), isStochastic = True), label="Bagging from the Past")
plt.title("Averaged Regret on the Partial-Information Stochastic Bernoulli Bandit problem (No Gap)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('FI No Gap Stochastic')
plt.close()

arms_mean[4] = 0.52

plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = UCB_factory(partial_info = True), isStochastic = True), label="Upper-Confidence-Bound")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = explore_then_commit_factory(5, partial_info = True), isStochastic = True), label="Explore-then-Commit")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True), isStochastic = True), label="Bagging from the Past")
plt.title("Averaged Regret on the Partial-Information Stochastic Bernoulli Bandit problem (Low Gap)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('FI Low Gap Stochastic')
plt.close()

arms_mean = np.random.uniform(low = 0.1, high = 0.5, size = K)
arms_mean[4] = 0.75

plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = UCB_factory(partial_info = True), isStochastic = True), label="Upper-Confidence-Bound")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = explore_then_commit_factory(5, partial_info = True), isStochastic = True), label="Explore-then-Commit")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3), label="Exp3")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = exp3_ix), label="Exp3-IX")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, get_arm_and_weight_to_pull = tsallis_inf), label="Tsallis-Inf")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True), isStochastic = True), label="Bagging from the Past")
plt.title("Averaged Regret on the Partial-Information Stochastic Bernoulli Bandit problem (High Gap)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('FI High Gap Stochastic')
plt.close()