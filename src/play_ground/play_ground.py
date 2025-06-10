from abc import ABC, abstractmethod
import matplotlib.pyplot as plt

class PlayGround:
    def __init__(self, data_generating_mechanism, games, plot_label, plot_directory):
        self.data_generating_mechanism = data_generating_mechanism
        self.games = games
        self.label = plot_label
        self.plot_directory = plot_directory

    def simulate_games(self):
        for i in range(len(self.games)):
            self.games[i].compute_averaged_regret()

    def plot_results(self):
        self.simulate_games()

        plt.rcParams["figure.figsize"] = (15,6)
        for i in range(len(self.games)):
            plt.plot(range(self.data_generating_mechanism.get_T()), 
                        self.games[i].get_instantaneuous_regret(), 
                        label = self.games[i].label)
        
        plt.legend()
        plt.title(self.label)
        plt.ylabel('Average Instantaneuous Regret')
        plt.xlabel('Round n')
        plt.savefig(self.plot_directory + self.label)
        plt.close()