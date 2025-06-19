from abc import ABC, abstractmethod
import matplotlib.pyplot as plt
import numpy as np

class PlayGround:
    def __init__(self, gap, games, hyperparameters, plot_label, plot_directory):
        self.games = games
        self.hyperparameters = hyperparameters
        self.label = plot_label
        self.plot_directory = plot_directory
        self.horizon = self.games[0].data_generating_mechanism.get_T()
        self.num_runs = self.games[0].data_generating_mechanism.get_M()
        self.gap = gap

    def plot_results(self):

        for i in range(len(self.hyperparameters)):
            ave_regret = np.zeros(shape = (len(self.hyperparameters[i]), 
                                                self.games[i].data_generating_mechanism.get_T()))
            med_regret = np.zeros(shape = (len(self.hyperparameters[i]), 
                                                self.games[i].data_generating_mechanism.get_T()))
            regret_final = np.zeros(shape = (self.games[i].data_generating_mechanism.get_M(),
                                             len(self.hyperparameters[i])))
            instantaneuous_regret = np.zeros(shape = (len(self.hyperparameters[i]), 
                                                self.games[i].data_generating_mechanism.get_T()))
            
            for j in range(len(self.hyperparameters[i])):
                hyper_parameter_i_j = self.hyperparameters[i][j]
                instantaneuous_regret[j, :] = self.games[i].get_instantaneuous_regret(hyper_parameter_i_j)
                ave_regret[j, :] = self.games[i].get_averaged_regret()
                regret_final[:, j] = self.games[i].get_regret_final()
                med_regret[j, :] = self.games[i].get_median_regret()

            plt.rcParams["figure.figsize"] = (15,6)
            for j in range(len(self.hyperparameters[i])):
                hyper_parameter_i_j = self.hyperparameters[i][j]
                plt.plot(range(self.horizon), 
                            instantaneuous_regret[j, :], 
                            label = "{} delta: {}".format(self.games[i].label, hyper_parameter_i_j))
            plt.legend()
            plt.title("Instantaneuous Regret (Gap = {}, T = {}, M = {})".format(self.gap, 
                                                                                self.horizon, 
                                                                                self.num_runs))
            plt.ylabel('Average Instantaneuous Regret')
            plt.xlabel('Round n')
            plt.savefig(self.plot_directory + "Instantaneuous_Regret")
            plt.close()

            plt.rcParams["figure.figsize"] = (15,6)
            for j in range(len(self.hyperparameters[i])):
                hyper_parameter_i_j = self.hyperparameters[i][j]
                plt.plot(range(self.horizon), 
                            ave_regret[j, :], 
                            label = "{} delta: {}".format(self.games[i].label, hyper_parameter_i_j))
        
            plt.legend()
            plt.title("Average Cumulative Regret (Gap = {}, T = {}, M = {})".format(self.gap, 
                                                                                    self.horizon, 
                                                                                    self.num_runs))
            plt.ylabel('Average Cumulative Regret')
            plt.xlabel('Round n')
            plt.savefig(self.plot_directory + "Cumulative_Regret")
            plt.close()

            plt.rcParams["figure.figsize"] = (15,6)
            for j in range(len(self.hyperparameters[i])):
                hyper_parameter_i_j = self.hyperparameters[i][j]
                plt.plot(range(self.horizon), 
                            med_regret[j, :], 
                            label = "{} delta: {}".format(self.games[i].label, hyper_parameter_i_j))
        
            plt.legend()
            plt.title("Median Cumulative Regret (Gap = {}, T = {}, M = {})".format(self.gap, 
                                                                                    self.horizon, 
                                                                                    self.num_runs))
            plt.ylabel('Median Cumulative Regret')
            plt.xlabel('Round n')
            plt.savefig(self.plot_directory + "Median_Cumulative_Regret")
            plt.close()

            plt.rcParams["figure.figsize"] = (15,6)
            fig, ax = plt.subplots()
            ax.set_ylabel('Regret_T')
            bplot = ax.boxplot(regret_final) 
            plt.title("Final Regret (Gap = {}, T = {}, M = {})".format(self.gap, self.horizon, self.num_runs))
            plt.savefig(self.plot_directory + "Final_Regret_Box_Plot")
            plt.close()
