import numpy as np
from numpy import random
import itertools
from sklearn.linear_model import LogisticRegression
from scipy.stats import norm
from scipy.optimize import root_scalar, minimize_scalar
from e2d.technical_tools.constants import HOEFFDING_SAMPLE_SIZE, SUBGAUSSIAN_SAMPLE_SIZE, ASYMPTOTIC_SAMPLE_SIZE, HELLINGER_SQUARE, MEAN_SQUARE, GAUSSIAN_MODELS, BERNOULLI_MODELS


# MC Estimator
class MC_Estimator():
    def __init__(self, finite_model_class):
        super().__init__()
        self.finite_model_class = finite_model_class
        self.K = self.finite_model_class.K
        self.M = self.finite_model_class.M
        self.combs = list(itertools.combinations(range(self.finite_model_class.M), 2))
        self.m_1 = -1
        self.m_2 = -1
    
    def clear(self, type):
        if type >= HOEFFDING_SAMPLE_SIZE:
            self.samples_drawn = np.zeros(shape = np.shape(self.samples_drawn))
        
    def draw_samples(self, m):
        # (M x K x m)
        self.m = m
        self.samples_drawn = np.zeros(shape = (self.finite_model_class.get_model_class_length(),
                                               self.finite_model_class.K, 
                                               self.m))
        
        for i in range(self.finite_model_class.get_model_class_length()):
            self.samples_drawn[i, :] = self.finite_model_class.draw_sample_from_model_index(i, self.m)

    def get_f_m_hat(self, type):
        if (type >= HOEFFDING_SAMPLE_SIZE):
            if (self.m_1 > -1):
                self.draw_samples(self.m_1)
            else:
                self.draw_samples(self.m)
            f_m_hat = np.average(self.samples_drawn, axis=2)
        else:
            f_m_hat = np.zeros(shape = (self.M, self.K))
            for i in range(self.M):
                f_m_hat[i, :] = self.finite_model_class.models[i].arm_means
        return f_m_hat

    def print_true_squared_hellinger_distance(self):
        true_sq_hellinger_divergence = np.zeros(shape = (len(self.combs), self.finite_model_class.K))
      
        #print("True Hellinger Divergence: ")
        for s in range(len(self.combs)):
            model_i_index = self.combs[s][0]
            model_j_index = self.combs[s][1]
            true_sq_hellinger_divergence[s, :] = self.finite_model_class.compute_true_sq_hellinger_divergence(model_i_index, model_j_index)
            #print("Model {0} Model {1} {2}".format(model_i_index, model_j_index, np.array2string(true_sq_hellinger_divergence[s, :])))

    def estimate_sq_hellinger_divergence(self, model_index_i, model_index_j):
        if (self.m_2 > -1):
            sample_model_i = self.finite_model_class.draw_sample_from_model_index(model_index_i, self.m_2)
            sample_model_j = self.finite_model_class.draw_sample_from_model_index(model_index_j, self.m_2)
            m = self.m_2
        else:
            sample_model_i = self.finite_model_class.draw_sample_from_model_index(model_index_i, self.m)
            sample_model_j = self.finite_model_class.draw_sample_from_model_index(model_index_j, self.m)
            m = self.m
        sq_hellinger_hat = np.zeros(shape = self.finite_model_class.K)

        for a in range(self.finite_model_class.K):
            model_i_action_a = sample_model_i[a, :]
            model_j_acton_a = sample_model_j[a, :]

            lambda_mixture = 0.5
            bootstrap_sample_x = []
            y = np.random.binomial(n = 1, p = lambda_mixture, size = m)
            y[0] = 0
            y[1] = 1
            for j in range(m):
                if (y[j] == 0):
                    bootstrap_sample_x.append([model_i_action_a[np.random.randint(m)]])
                else:
                    bootstrap_sample_x.append([model_j_acton_a[np.random.randint(m)]])

            clf = LogisticRegression(random_state=0, 
                                     solver='liblinear',
                                     max_iter=500).fit(bootstrap_sample_x, y)

            y = np.random.binomial(n = 1, p = lambda_mixture, size = m)
            for j in range(m):
                if (y[j] == 0):
                    bootstrap_sample_x.append([model_i_action_a[np.random.randint(m)]])
                else:
                    bootstrap_sample_x.append([model_j_acton_a[np.random.randint(m)]])

            prediction_prob = clf.predict_proba(bootstrap_sample_x)
            sq_hellinger_hat_mc = np.zeros(shape=self.m)
            for j in range(self.m):
                sq_hellinger_hat_mc[j] = (np.sqrt(prediction_prob[j][0] / (1 - lambda_mixture)) - np.sqrt(prediction_prob[j][1] / (lambda_mixture)))**2
            sq_hellinger_hat[a] = np.mean(sq_hellinger_hat_mc)
        return sq_hellinger_hat 
    
    def estimated_radon_nikodym_derivative(self, model_index_i, model_index_j, x):
        print("Estimating with sampe size {}".format(self.m))
        derivatives = np.zeros(shape = (2, self.K, len(x)))
        sample_model_i = self.finite_model_class.draw_sample_from_model_index(model_index_i, self.m)
        sample_model_j = self.finite_model_class.draw_sample_from_model_index(model_index_j, self.m)

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
                    bootstrap_sample_x.append([self.finite_model_class.draw_sample_from_model_index(model_index_i, 1)[a, 0]])
                else:
                    bootstrap_sample_x.append([self.finite_model_class.draw_sample_from_model_index(model_index_j, 1)[a, 0]])

            clf = LogisticRegression(random_state=0, 
                                     solver='liblinear',
                                     max_iter=500).fit(bootstrap_sample_x, y)

            y = np.random.binomial(n = 1, p = lambda_mixture, size = self.m)
            for j in range(len(x)):
                bootstrap_sample_x.append([x[j]])

            prediction_prob = clf.predict_proba(bootstrap_sample_x)

            for j in range(len(x)):
                derivatives[1][a][j] = prediction_prob[j][0] / (1 - lambda_mixture)
                derivatives[0][a][j] = prediction_prob[j][1] / (lambda_mixture)
            
        return derivatives

    
    def estimate_mean_square_divergence(self,  model_index_i, model_index_j):
        sample_model_i = self.finite_model_class.draw_sample_from_model_index(model_index_i, self.m)
        sample_model_j = self.finite_model_class.draw_sample_from_model_index(model_index_j, self.m)
        sq_hellinger_hat = np.zeros(shape = self.finite_model_class.K)

        for a in range(self.finite_model_class.K):
            model_i_action_a = sample_model_i[a, :]
            model_j_acton_a = sample_model_j[a, :]
            sq_hellinger_hat_mc = np.zeros(shape=self.m)
            for j in range(self.m):
                sq_hellinger_hat_mc[j] = (model_i_action_a[j] - model_j_acton_a[j])**2
            sq_hellinger_hat[a] = np.mean(sq_hellinger_hat_mc)
        return sq_hellinger_hat
    
    def get_divergence_hat(self, sample_size_type, divergence_type):
        self.sq_hellinger_divergence_hat = np.zeros(shape = (len(self.combs), self.finite_model_class.K))
        for s in range(len(self.combs)):
            model_index_i = self.combs[s][0]
            model_index_j = self.combs[s][1]
            if (divergence_type == MEAN_SQUARE):
                if (sample_size_type >= HOEFFDING_SAMPLE_SIZE):
                    self.sq_hellinger_divergence_hat[s, :] = self.estimate_mean_square_divergence(model_index_i, model_index_j)
                else:
                    self.sq_hellinger_divergence_hat[s, :] = self.finite_model_class.compute_true_mean_square_divergence(model_index_i, model_index_j)
            else:
                if (sample_size_type >= HOEFFDING_SAMPLE_SIZE):
                    self.sq_hellinger_divergence_hat[s, :] = self.estimate_sq_hellinger_divergence(model_index_i, model_index_j)
                else:
                    self.sq_hellinger_divergence_hat[s, :] = self.finite_model_class.compute_true_sq_hellinger_divergence(model_index_i, model_index_j)
        return self.sq_hellinger_divergence_hat
    
    def get_bias_of_divergence_hat(self, divergence_type):
        self.bias_divergence_hat = np.zeros(shape = (len(self.combs), self.finite_model_class.K))
        self.true_divergence = np.zeros(shape = (len(self.combs), self.finite_model_class.K))
        for s in range(len(self.combs)):
            model_index_i = self.combs[s][0]
            model_index_j = self.combs[s][1]
            if (divergence_type == HELLINGER_SQUARE):
                estimated_divergence_s = self.estimate_sq_hellinger_divergence(model_index_i=model_index_i, 
                                                                                  model_index_j=model_index_j)
                true_divergence_s = self.finite_model_class.compute_true_sq_hellinger_divergence(model_index_i, 
                                                                                                    model_index_j)
            else:
                estimated_divergence_s = self.estimate_mean_square_divergence(model_index_i=model_index_i, 
                                                                                 model_index_j=model_index_j)
                true_divergence_s = self.finite_model_class.compute_true_mean_square_divergence(model_index_i, 
                                                                                                   model_index_j)
                
            self.bias_divergence_hat[s, :] = estimated_divergence_s - true_divergence_s
            self.true_divergence[s, :] = true_divergence_s

        bias_hat_flat = self.bias_divergence_hat.flatten()
        sq_hellinger_flat = self.true_divergence.flatten()
        
        return [np.mean(bias_hat_flat),
                np.mean(sq_hellinger_flat),
                np.std(bias_hat_flat),
                bias_hat_flat]
    
    def get_squared_hellinger_distance_lower_bound(self, optimality_gap, sigma_p=0.5, sigma_q=0.5):
        return (1 - np.sqrt(1 - (optimality_gap**2 / (optimality_gap**2 + (sigma_p + sigma_q)**2))))
    
    def mean_objective_func(self, m, beta):
        return ((2 * self.M * self.K * (1 - norm.cdf(np.sqrt(m) * self.optimality_gap / (2 * 0.5)))) - (self.delta * beta))

    def hellinger_objective_func(self, m, beta):
        var_upper_bound = np.sqrt(4 - self.get_squared_hellinger_distance_lower_bound(self.optimality_gap))
        return (2 * self.M * (self.M - 1) * self.K * (1 - norm.cdf((np.sqrt(m) * self.delta_delta)/(2 * (np.sqrt(var_upper_bound)))))) - (self.delta * (1 - beta)) 
    
    def hoeffding_mc_estimate(self, beta, b = 1, a = 0):
        return 2 * (((b - a) / self.optimality_gap)**2) * np.log(2 * self.M * self.K / (self.delta * beta))

    def subgaussian_mc_estimate(self, beta, sigma = 0.5):
        return 2 * ((sigma / self.optimality_gap)**2) * np.log(2 * self.M * self.K / (self.delta * beta))

    def asymptotic_mc_estimate(self, beta):
        return int(root_scalar(self.mean_objective_func, args = beta, method = "brentq", bracket=[0, 1000000000]).root)

    def hoeffding_hellinger_estimate(self, beta):
        d_2_h = self.delta_delta
        return (8 / d_2_h**2) * np.log((self.K * self.M * (self.M - 1)) / ((self.delta) * (1 - beta)))

    def asymptotic_hellinger_estimate(self, beta):
        return int(root_scalar(self.hellinger_objective_func, args = beta, method="brentq", bracket=[0, 1000000000]).root)
                         
    def max_func(self, beta):
        sample_size_m1 = self.M * self.K * self.func_1(beta)
        sample_size_m2 =  ((self.M * (self.M - 1) * self.K) / 2) * self.func_2(beta)
        return sample_size_m1 + sample_size_m2

    def compute_sample_size_m(self, delta, type):
        self.optimality_gap = self.finite_model_class.get_delta_min()
        self.delta_delta = self.finite_model_class.get_delta_delta_min()
        self.delta = delta
        self.delta_delta = self.delta_delta
        
        if (type >= HOEFFDING_SAMPLE_SIZE):
            if (type == HOEFFDING_SAMPLE_SIZE):
                self.func_1 = self.hoeffding_mc_estimate
                self.func_2 = self.hoeffding_hellinger_estimate
                res = minimize_scalar(self.max_func, bounds=(0, 1), method='bounded')
                self.beta = res.x
                self.m_1 = int(np.ceil(self.func_1(self.beta)))
                self.m_2 = int(np.ceil(self.func_2(self.beta)))
                self.m_2 = np.min([1000000, self.m_2])
                print("MC sample size for controlling prob. of bad event with small delta {0} on model gaps {1}, {2} is {3}, {4}"
                      .format(delta, self.optimality_gap, self.delta_delta, self.m_1, self.m_2))
            elif (type == SUBGAUSSIAN_SAMPLE_SIZE):
                self.func_1 = self.subgaussian_mc_estimate
                self.func_2 = self.hoeffding_hellinger_estimate
                res = minimize_scalar(self.max_func, bounds=(0, 1), method='bounded')
                self.beta = res.x
                self.m_1 = int(np.ceil(self.func_1(self.beta)))
                self.m_2 = int(np.ceil(self.func_2(self.beta)))
                self.m_2 = np.min([1000000, self.m_2])
                print("Subgaussian")
                print("MC sample size for controlling prob. of bad event with small delta {0} on model gaps {1}, {2} is {3}, {4}"
                      .format(delta, self.optimality_gap, self.delta_delta, self.m_1, self.m_2))
            elif (type == ASYMPTOTIC_SAMPLE_SIZE):
                self.func_1 = self.asymptotic_mc_estimate
                self.func_2 = self.asymptotic_hellinger_estimate
                res = minimize_scalar(self.max_func, bounds=(0, 1), method='bounded')
                self.beta = res.x
                self.m_1 = int(np.ceil(self.func_1(self.beta)))
                self.m_2 = int(np.ceil(self.func_2(self.beta)))
                self.m_2 = np.min([1000000, self.m_2])
                print("Asymptotic")
                print("MC sample size for controlling prob. of bad event with small delta {0} on model gaps {1}, {2} is {3}, {4}"
                      .format(delta, self.optimality_gap, self.delta_delta, self.m_1, self.m_2))
            else:
                self.m = type
        else:
            self.m = -1
        return -1