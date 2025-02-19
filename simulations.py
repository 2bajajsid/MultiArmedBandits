import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from itertools import chain
from scipy.optimize import Bounds, LinearConstraint, root_scalar
import math

arms_mean = np.random.uniform(low = 0.1, high = 0.5, size = K)
arms_mean[4] = 0.75

plt.rcParams["figure.figsize"] = (15,6)
""" plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                    get_arm_to_pull = bagging_from_past_v_old_factory(5)), 
                                                    label="b_from_past_v1")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                    get_arm_to_pull = bagging_from_past_into_future_factory(5)), 
                                                    label="b_from_past_v2")
plt.plot(range(T), simulate_full_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                    get_arm_to_pull = dropout_factory(5, prob=0.05)), 
                                                    label="DropOut (0.05)")
plt.legend()
plt.title("Averaged Regret on the Full-Information Stochastic Bernoulli Bandit problem (Delta > 0.25)")
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('FI High-Gap Stochastic Problem')
plt.close() """
 
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit,
                                                        get_arm_and_weight_to_pull = bagging_from_past_v_old_factory(5, partial_info=True)), 
                                                        label="b_from_past_v1")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = bagging_from_past_into_future_factory(5, partial_info=True)), 
                                                        label="b_from_past_v2")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = dropout_factory(5, partial_info=True, prob=0.05)), 
                                                        label="DropOut (0.05)")
plt.plot(range(T), simulate_partial_information_problem(K = K, T = T, generate_rewards = bernoulli_bandit, 
                                                        get_arm_and_weight_to_pull = exp3_plus_plus), 
                                                        label="Exp3++")
plt.title("Averaged Regret on the Partial-Information Stochastic Bernoulli Bandit problem (Delta > 0.25)")

plt.legend()
plt.ylabel('Regret')
plt.xlabel('Round n')
plt.savefig('PI High-Gap Stochastic Problem')
plt.close()