import numpy as np
from numpy import random
import itertools
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize_scalar

# MC Estimator
class MC_Estimator():
    def __init__(self, finite_model_class):
        super().__init__()
        self.finite_model_class = finite_model_class
        self.K = self.finite_model_class.K
        self.M = self.finite_model_class.M
        self.combs = list(itertools.combinations(range(self.finite_model_class.M), 2))
    
    def clear(self):
        self.samples_drawn = np.zeros(shape = (self.finite_model_class.get_model_class_length(),
                                               self.finite_model_class.K,
                                               self.m))
        
    def draw_samples(self, m):
        # (M x K x m)
        self.m = m
        self.samples_drawn = np.zeros(shape = (self.finite_model_class.get_model_class_length(),
                                               self.finite_model_class.K, 
                                               self.m))
        
        for i in range(self.finite_model_class.get_model_class_length()):
            self.samples_drawn[i, :] = self.finite_model_class.draw_sample_from_model_index(i, self.m)

    def get_f_m_hat(self):
        f_m_hat = np.average(self.samples_drawn, axis=2)
        #print("Mean Estimates {} with sample size {}".format(f_m_hat, self.m))
        return f_m_hat

    def print_true_squared_hellinger_distance(self):
        true_sq_hellinger_divergence = np.zeros(shape = (len(self.combs), self.finite_model_class.K))
      
        #print("True Hellinger Divergence: ")
        for s in range(len(self.combs)):
            model_i_index = self.combs[s][0]
            model_j_index = self.combs[s][1]
            true_sq_hellinger_divergence[s, :] = self.finite_model_class.compute_true_sq_hellinger_divergence(model_i_index, model_j_index)
            #print("Model {0} Model {1} {2}".format(model_i_index, model_j_index, np.array2string(true_sq_hellinger_divergence[s, :])))

    def estimate_sq_hellinger_divergence(self, s):
        model_index_i = self.combs[s][0]
        model_index_j = self.combs[s][1]

        sample_model_i = self.samples_drawn[model_index_i, :]
        sample_model_j = self.samples_drawn[model_index_j, :]
        sq_hellinger_hat = np.zeros(shape = self.finite_model_class.K)

        for a in range(self.finite_model_class.K):
            model_i_action_a = sample_model_i[a, :]
            model_j_acton_a = sample_model_j[a, :]

            lambda_mixture = 0.5
            bootstrap_sample_x = []
            y = np.random.binomial(n = 1, p = lambda_mixture, size = self.m)
            y[0] = 0
            y[1] = 1
            for j in range(self.m):
                if (y[j] == 0):
                    bootstrap_sample_x.append([model_i_action_a[np.random.randint(self.m)]])
                else:
                    bootstrap_sample_x.append([model_j_acton_a[np.random.randint(self.m)]])

            clf = LogisticRegression(random_state=0, 
                                     solver='liblinear',
                                     max_iter=500).fit(bootstrap_sample_x, y)

            y = np.random.binomial(n = 1, p = lambda_mixture, size = self.m)
            for j in range(self.m):
                if (y[j] == 0):
                    bootstrap_sample_x.append([model_i_action_a[np.random.randint(self.m)]])
                else:
                    bootstrap_sample_x.append([model_j_acton_a[np.random.randint(self.m)]])

            prediction_prob = clf.predict_proba(bootstrap_sample_x)
            sq_hellinger_hat_mc = np.zeros(shape=self.m)
            for j in range(self.m):
                sq_hellinger_hat_mc[j] = (np.sqrt(prediction_prob[j][0] / (1 - lambda_mixture)) - np.sqrt(prediction_prob[j][1] / (lambda_mixture)))**2
            sq_hellinger_hat[a] = np.mean(sq_hellinger_hat_mc)
        return sq_hellinger_hat
    
    def estimate_mean_square_divergence(self, s):
        model_index_i = self.combs[s][0]
        model_index_j = self.combs[s][1]

        sample_model_i = self.samples_drawn[model_index_i, :]
        sample_model_j = self.samples_drawn[model_index_j, :]
        sq_hellinger_hat = np.zeros(shape = self.finite_model_class.K)

        for a in range(self.finite_model_class.K):
            model_i_action_a = sample_model_i[a, :]
            model_j_acton_a = sample_model_j[a, :]
            sq_hellinger_hat_mc = np.zeros(shape=self.m)
            for j in range(self.m):
                sq_hellinger_hat_mc[j] = (model_i_action_a[j] - model_j_acton_a[j])**2
            sq_hellinger_hat[a] = np.mean(sq_hellinger_hat_mc)
        return sq_hellinger_hat

    def get_sq_hellinger_divergence_hat(self):
        self.sq_hellinger_divergence_hat = np.zeros(shape = (len(self.combs), self.finite_model_class.K))
        for s in range(len(self.combs)):
            self.sq_hellinger_divergence_hat[s, :] = self.estimate_sq_hellinger_divergence(s)

        # print("Estimated Hellinger Divergence: ")
        # for s in range(len(self.combs)):
        #    model_i_index = self.combs[s][0]
        #    model_j_index = self.combs[s][1]
        #       print("Model {0} Model {1} {2}".format(model_i_index, model_j_index, np.array2string(self.sq_hellinger_divergence_hat[s, :])))
        return self.sq_hellinger_divergence_hat
    
    def get_squared_hellinger_distance_lower_bound(self, optimality_gap, sigma_p=0.5, sigma_q=0.5):
        return (1 - np.sqrt(1 - (optimality_gap**2 / (optimality_gap**2 + (sigma_p + sigma_q)**2))))

    def get_m_for_bernoulli_model(self, delta, optimality_gap):
        sq_hellinger_distance_lower_bound = self.get_squared_hellinger_distance_lower_bound(optimality_gap, sigma_p = 1, sigma_q = 1)
        print("Sq_Hellinger_Lower_Bound: {0}".format(sq_hellinger_distance_lower_bound))
        hellinger_mc_estimate = int((32 * (self.K / sq_hellinger_distance_lower_bound)**2) * np.log(self.K * self.M * (self.M - 1) / delta))
        mean_mc_estimate = int((8 * (self.K / optimality_gap)**2) * np.log(2 * self.K * self.M / delta))
        m = np.max([hellinger_mc_estimate, mean_mc_estimate])
        m=10
        print("Hellinger MC Estimate: {0}".format(hellinger_mc_estimate))
        print("Mean MC Estimate: {0}".format(mean_mc_estimate))
        print("MC sample size for controlling prob. of bad event with small delta {0} on model gap {1} is {2}".format(delta, optimality_gap, m))
        return m
    
    def mean_objective_func(self, m, beta):
        return ((2 * self.M * self.K * (1 - norm.cdf(np.sqrt(m) * self.optimality_gap / 2))) - (self.delta * beta))

    def hellinger_objective_func(self, m, beta):
        var_upper_bound = np.sqrt(4 - self.get_squared_hellinger_distance_lower_bound(self.delta_delta))
        return (2 * self.M * (self.M - 1) * self.K * (1 - norm.cdf((np.sqrt(m) * self.delta_delta)/(2 * (np.sqrt(var_upper_bound)))))) - (self.delta * (1 - beta)) 
    
    def func_1(self, beta, b = 1, a = -1):
        return 2 * (((b - a) / self.optimality_gap)**2) * np.log(2 * self.M * self.K / (self.delta * beta))
        #return ((4 * self.M * self.K) / (beta * self.delta * self.optimality_gap**2))
        return int(root_scalar(self.mean_objective_func, args = beta, method = "brentq", bracket=[0, 1000000]).root)
        
    def func_2(self, beta):
        d_2_h = self.get_squared_hellinger_distance_lower_bound(self.delta_delta)
        return (8 / d_2_h) * np.log((self.K * self.M * (self.M - 1)) / ((self.delta) * (1 - beta)))
        #return ((2 * self.M * self.K) * (self.M - 1) * (4 - d_2_h)) / (self.delta * (1 - beta) * self.optimality_gap**2)
        #return int(root_scalar(self.hellinger_objective_func, args = beta, method="brentq", bracket=[0, 1000000]).root)
                                    
    def max_func(self, beta):
        return np.max([self.func_1(beta), 
                       self.func_2(beta)])

    def get_m_for_gaussian_model(self, delta):
        self.optimality_gap = self.finite_model_class.get_delta_min()
        self.delta_delta = self.optimality_gap
        sq_hellinger_distance_lower_bound = self.get_squared_hellinger_distance_lower_bound(self.optimality_gap, sigma_p = 0.5, sigma_q = 0.5)
        self.delta = delta
        print("Sq_Hellinger_Lower_Bound: {0}".format(sq_hellinger_distance_lower_bound))
        # Hoeffding's 
        #hellinger_mc_estimate = int(32 * ((1 / delta_delta)**2) * np.log(2 * self.K * self.M * (self.M - 1) / delta))
        #mean_mc_estimate = int((2 * (2 / optimality_gap)**2) * np.log(4 * self.K * self.M / delta))
        beta = minimize_scalar(self.max_func, bounds=(0, 1), method='bounded').x
        print("beta: {}".format(beta))
        hellinger_mc_estimate = int(self.func_2(beta))
        mean_mc_estimate = int(self.func_1(beta))
        # Chebyshev's 
        #hellinger_mc_estimate = int(((8 * self.M * self.K) * (self.M - 1) * (4 - sq_hellinger_distance_lower_bound)) / (delta * optimality_gap**2))
        #mean_mc_estimate = int((8 * self.M * self.K) / (delta * optimality_gap**2))
        # Asymptotics
        #mean_mc_estimate = int(root_scalar(self.mean_objective_func, 
        #                               method = "brentq", bracket=[0, 1000000]).root)
        #hellinger_mc_estimate = int(root_scalar(self.hellinger_objective_func, 
        #                                method="brentq", bracket=[0, 1000000]).root)
        m = np.max([hellinger_mc_estimate, mean_mc_estimate])
        #m = 10
        print("Hellinger MC Estimate: {0}".format(hellinger_mc_estimate))
        print("Mean MC Estimate: {0}".format(mean_mc_estimate))
        print("MC sample size for controlling prob. of bad event with small delta {0} on model gaps {1}, {2} is {3}".format(delta, self.optimality_gap, self.delta_delta, m))
        return m