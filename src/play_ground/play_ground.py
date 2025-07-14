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

    def plot_regret_as_function_of_hyperparameters(self, vlines):
        regret_final = np.zeros(shape=len(self.hyperparameters[0]))
        sd_final = np.zeros(shape=len(self.hyperparameters[0]))
        for j in range(len(self.hyperparameters[0])):
            regret_final[j] = self.games[0].compute_averaged_regret(self.hyperparameters[0][j])
            sd_final[j] = self.games[0].get_sd_final()
            self.hyperparameters[0][j] = np.log(1 / self.hyperparameters[0][j]['delta'])

        plt.rcParams["figure.figsize"] = (15,6)
        plt.ylabel('Average Final Regret')
        plt.xlabel('ln(1 / delta)')
        plt.title("Regret as a function of delta")
        plt.vlines(x=np.log(np.reciprocal(vlines[0])), ymin=0, ymax=np.max(regret_final), colors='purple', ls='--', lw=2, label='Tuned delta')
        plt.vlines(x=np.log(np.reciprocal(vlines[1])), ymin=0, ymax=np.max(regret_final), colors='green', ls='--', lw=2, label='1/T')
        plt.errorbar(self.hyperparameters[0], regret_final, yerr= sd_final, label = "Averaged Final Regret")
        plt.legend()
        plt.savefig(self.plot_directory + self.label + "_Regret_Curve")
        plt.close()

    def plot_regret_as_function_of_hyperparameters_2(self, vlines, del_vals):
        n = int(len(self.hyperparameters[0]))
        n_over_2 = int(n/3)
        regret_final = np.zeros(shape=(3, n_over_2))
        sd_final = np.zeros(shape=(3, n_over_2))
        hp_final = np.zeros(shape=(3, n_over_2))
        for k in range(3):
            for j in range(n_over_2):
                regret_final[k][j] = self.games[0].compute_averaged_regret(self.hyperparameters[0][k*n_over_2 + j])
                sd_final[k][j] = self.games[0].get_sd_final()
                hp_final[k][j] = self.hyperparameters[0][k*n_over_2 + j]['reward_sd']

        print("Done!")
        print(regret_final[0])
        print(regret_final[1])
        print(regret_final[2])

        plt.rcParams["figure.figsize"] = (15,6)
        plt.ylabel('Average Final Regret')
        plt.xlabel('Reward sd.')
        plt.title("Regret as a function of Reward sd.")
        plt.errorbar(hp_final[0], regret_final[0], yerr = sd_final[0], label = "Averaged Final Regret (Delta: {})".format(del_vals[0]))
        plt.errorbar(hp_final[1], regret_final[1], yerr = sd_final[1], label = "Averaged Final Regret (Delta: {})".format(del_vals[1]))
        plt.errorbar(hp_final[2], regret_final[2], yerr = sd_final[2], label = "Averaged Final Regret (Delta: {})".format(del_vals[2]))
        plt.legend()
        plt.savefig(self.plot_directory + self.label + "_Average_Regret_Curve")
        plt.close()

    def plot_regret_as_function_of_hyperparameters_3(self, p_cov):
        regret_final = np.zeros(shape=len(self.hyperparameters[0]))
        regret_median = np.zeros(shape=len(self.hyperparameters[0]))
        sd_final = np.zeros(shape=len(self.hyperparameters[0]))
        for j in range(len(self.hyperparameters[0])):
            regret_final[j] = self.games[0].compute_averaged_regret(self.hyperparameters[0][j])
            regret_median[j] = self.games[0].get_median_final_regret()
            self.hyperparameters[0][j] = self.hyperparameters[0][j]['lambda']
            sd_final[j] = self.games[0].get_sd_final()

        plt.rcParams["figure.figsize"] = (15,6)
        plt.ylabel('Average Final Regret')
        plt.xlabel('Lambda')
        plt.title("Regret as a function of penatly term lambda with posterior variance {}".format(p_cov))
        plt.errorbar(self.hyperparameters[0], regret_final, yerr=sd_final, label = "Averaged Final Regret")
        plt.legend()
        plt.savefig(self.plot_directory + self.label + "_Average_Regret_Curve")
        plt.close()

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
