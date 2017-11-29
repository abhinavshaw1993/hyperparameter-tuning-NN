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
"""
def scale(learning_rate_range, momentum_range,r):
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
    # Comverting to Foat.
    # momentum_range = float(momentum_range)
    step_size_w = momentum_range[1]-momentum_range[0]
    sample1 = data[0]
    print ("step_size ,sample1: ", step_size_w, sample1, step_size_w*sample1[1] / width)


scale(10,10,4)
