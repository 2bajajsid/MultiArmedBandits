from full_information_game import Full_Information_Game
import matplotlib.pyplot as plt
from play_ground import PlayGround

class Full_Info_Play_Ground(PlayGround):
    def __init__(self, data_generating_mechanism, bandit_algorithms, plot_label):
        games = []
        for i in range(len(bandit_algorithms)):
            games.append(Full_Information_Game(data_generating_mechanism=data_generating_mechanism, 
                                   bandit_algorithm= bandit_algorithms[i]))
        super().__init__(data_generating_mechanism = data_generating_mechanism, 
                        games = games, 
                        plot_label=plot_label)


    
