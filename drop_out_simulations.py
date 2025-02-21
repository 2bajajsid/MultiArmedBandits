import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from high_gap_stochastic import High_Gap_Stochastic
from stochastically_constrained import Stochastically_Constrained
from partial_info_play_ground import Partial_Info_Play_Ground
from full_info_play_ground import Full_Info_Play_Ground
from dropout import DropOut_FI, DropOut_PI
import math
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()

drop_out_point_zero_one = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.05)
drop_out_point_one = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.1)
drop_out_point_two_five = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.25)
drop_out_point_five = DropOut_FI(stochastically_constrained_data_job, dropout_prob=0.5)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [drop_out_point_zero_one, drop_out_point_zero_five,  
                                          drop_out_point_one, drop_out_point_two_five, drop_out_point_five], 
                                          plot_label="DropOut Full Information Stochastic Adversarial")
full_info_ground.plot_results()

drop_out_point_zero_one = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.05)
drop_out_point_one = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.1)
drop_out_point_two_five = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.25)
drop_out_point_five = DropOut_PI(stochastically_constrained_data_job, dropout_prob=0.5)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                               [drop_out_point_zero_one, drop_out_point_zero_five,  
                                                drop_out_point_one, drop_out_point_two_five, drop_out_point_five], 
                                               plot_label="DropOut Partial Information Stochastic Adversarial")
partial_info_ground.plot_results()

drop_out_point_zero_one = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.01)
drop_out_point_zero_five = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.05)
drop_out_point_one = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.1)
drop_out_point_two_five = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.25)
drop_out_point_five = DropOut_PI(high_gap_stochastic_data_job, dropout_prob=0.5)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [drop_out_point_zero_one, drop_out_point_zero_five,  
                                                drop_out_point_one, drop_out_point_two_five, drop_out_point_five], 
                                               plot_label="DropOut Partial Information High-Gap Stochastic")
partial_info_ground.plot_results()

