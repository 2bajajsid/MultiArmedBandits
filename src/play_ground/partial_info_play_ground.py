from game.partial_information_game import Partial_Information_Game
import matplotlib.pyplot as plt
from play_ground.play_ground import PlayGround

class Partial_Info_Play_Ground(PlayGround):
    def __init__(self, hyperparameters, bandit_algorithms, plot_label, plot_directory):
        games = []
        for i in range(len(bandit_algorithms)):
            games.append(Partial_Information_Game(
                                   bandit_algorithm= bandit_algorithms[i]
                        ))
        super().__init__(games=games,
                        hyperparameters=hyperparameters,
                        plot_label=plot_label,
                        plot_directory=plot_directory)