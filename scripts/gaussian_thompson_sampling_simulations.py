import numpy as np
import matplotlib.pyplot as plt
from numpy import random
from high_gap_stochastic import High_Gap_Stochastic
from stochastically_constrained import Stochastically_Constrained
from src.play_ground.partial_info_play_ground import Partial_Info_Play_Ground
from full_info_play_ground import Full_Info_Play_Ground
from gaussian_thompson_sampling import Gaussian_Thompson_FI, Gaussian_Thompson_PI
import math
import numpy as np

high_gap_stochastic_data_job = High_Gap_Stochastic()
stochastically_constrained_data_job = Stochastically_Constrained()

gt_K = Gaussian_Thompson_FI(stochastically_constrained_data_job, stochastically_constrained_data_job.get_K())
gt_K_half = Gaussian_Thompson_FI(stochastically_constrained_data_job, stochastically_constrained_data_job.get_K() / 2)
gt_K_sqrt = Gaussian_Thompson_FI(stochastically_constrained_data_job, math.sqrt(stochastically_constrained_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_FI(stochastically_constrained_data_job, 0.01)
full_info_ground = Full_Info_Play_Ground(stochastically_constrained_data_job, 
                                         [gt_K, gt_K_half, gt_K_sqrt, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                          plot_label="GT Full Information Stochastic Adversarial")
full_info_ground.plot_results()

gt_K = Gaussian_Thompson_PI(stochastically_constrained_data_job, stochastically_constrained_data_job.get_K())
gt_K_half = Gaussian_Thompson_PI(stochastically_constrained_data_job, stochastically_constrained_data_job.get_K() / 2)
gt_K_sqrt = Gaussian_Thompson_PI(stochastically_constrained_data_job, math.sqrt(stochastically_constrained_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_PI(stochastically_constrained_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_PI(stochastically_constrained_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_PI(stochastically_constrained_data_job, 0.01)
partial_info_ground = Partial_Info_Play_Ground(stochastically_constrained_data_job, 
                                               [gt_K, gt_K_half, gt_K_sqrt, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                               plot_label="GT Partial Information Stochastic Adversarial")
partial_info_ground.plot_results()

gt_K = Gaussian_Thompson_PI(high_gap_stochastic_data_job, stochastically_constrained_data_job.get_K())
gt_K_half = Gaussian_Thompson_PI(high_gap_stochastic_data_job, stochastically_constrained_data_job.get_K() / 2)
gt_K_sqrt = Gaussian_Thompson_PI(high_gap_stochastic_data_job, math.sqrt(stochastically_constrained_data_job.get_K()))
gt_K_1 = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 1)
gt_K_point_1 = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 0.1)
gt_K_point_zero_1 = Gaussian_Thompson_PI(high_gap_stochastic_data_job, 0.01)
partial_info_ground = Partial_Info_Play_Ground(high_gap_stochastic_data_job, 
                                               [gt_K, gt_K_half, gt_K_sqrt, gt_K_1, gt_K_point_1, gt_K_point_zero_1], 
                                               plot_label="GT Partial Information High-Gap Stochastic")
partial_info_ground.plot_results()