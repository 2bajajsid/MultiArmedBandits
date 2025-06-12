from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class PlayGround:
    def __init__(self, games, hyperparameters, plot_label, plot_directory):
        self.games = games
        self.hyperparameters = hyperparameters
        self.label = plot_label
        self.plot_directory = plot_directory
        self.horizon = self.games[0].data_generating_mechanism.get_T()
        self.num_runs = self.games[0].data_generating_mechanism.get_M()

    def plot_results(self):
        plt.rcParams["figure.figsize"] = (15,6)
        for i in range(len(self.hyperparameters)):
            for j in range(len(self.hyperparameters[i])):
                hyper_parameter_i_j = self.hyperparameters[i][j]
                plt.plot(range(self.horizon), 
                            self.games[i].get_instantaneuous_regret(hyper_parameter_i_j), 
                            label = "{} hyper_parameter {}".format(self.games[i].label, j))
        
        plt.legend()
        plt.title(self.label)
        plt.ylabel('Average Instantaneuous Regret')
        plt.xlabel('Round n')
        plt.savefig(self.plot_directory + self.label)
        plt.close()

        self.plot_box_plot()

    def plot_box_plot(self):
        plt.rcParams["figure.figsize"] = (15,6)

        regret_stats = np.zeros(shape = (self.num_runs, len(self.games)))
        for i in range(len(self.games)):
            regret_stats[:, i] = self.games[i].get_regret_final()
        
        fig, ax = plt.subplots()
        ax.set_ylabel('Regret_T')
        bplot = ax.boxplot(regret_stats) 
        plt.savefig(self.plot_directory + self.label + "boxplot")
        plt.close()
