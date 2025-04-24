from game.partial_information_game import Partial_Information_Game
import matplotlib.pyplot as plt
from play_ground.play_ground import PlayGround

class Partial_Info_Play_Ground(PlayGround):
    def __init__(self, data_generating_mechanism, bandit_algorithms, plot_label, plot_directory, compute_importance_weighted_rewards = False):
        games = []
        for i in range(len(bandit_algorithms)):
            games.append(Partial_Information_Game(data_generating_mechanism=data_generating_mechanism, 
                                   bandit_algorithm= bandit_algorithms[i], 
                                   compute_importance_weighted_rewards=compute_importance_weighted_rewards))
        super().__init__(data_generating_mechanism=data_generating_mechanism, 
                        games=games,
                        plot_label=plot_label,
                        plot_directory=plot_directory)