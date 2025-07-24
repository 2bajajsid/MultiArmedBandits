import numpy as np
from numpy import random
import itertools
from sklearn.linear_model import LogisticRegression

# MC Estimator
class MC_Estimator():
    def __init__(self, finite_model_class):
        super().__init__()
        self.finite_model_class = finite_model_class
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