###############################################################################
# This module will scale the Hyperparameter range and sample using Poisson Disc
# Sampling.
###############################################################################

import numpy as np
from  poisson_disc import *

"""
Input:
- learning_rate_range: Search Range for Learning Rate.
- momentum_range: Search Range for momentum_range.
- r: The Threshold radius of samples. (Density of samples increases as r
  decreseas, so does time taken.)

Returns:
- hyperparameters: List of Tuples. Tuple Stucture -(learning_rate, momentum).
- sample_count: Number of samples returned.
"""
def scaler(learning_rate_range, momentum_range,r):
    # Initializing length and width. Can Be increased fo changing number of samples.
    length = 20
    width = 10

    # Sample Count.
    size = r / sqrt(2)
    cells = (length//size)* (width//size)

    # Initilizing Poisson Disc Sampling object.
    grid = Grid(r, length, width)

    # Random Seed.
    rand = (random.uniform(0, length), random.uniform(0, width))
    data = grid.poisson(rand)

    print ("Data: ", data)

    momentum_range = (0.0,1.0)
    learning_rate_range = (1e-5, 1e-1)

    step_size_l = learning_rate_range[1] - learning_rate_range[0]
    step_size_w = momentum_range[1]-momentum_range[0]
    hyperparameters = []

    # Converting Poisson Samples to Hyperparameter Samples.
    for i in range(0,len(data)):
        sample = data[i]
        learning_rate, momentum = step_size_l*sample[0] / length, step_size_w*sample[1] / width
        hyperparameters.append((learning_rate, momentum))

    print (hyperparameters)

scaler(10,10,4)
