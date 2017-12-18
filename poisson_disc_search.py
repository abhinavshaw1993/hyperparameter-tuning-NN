###############################################################################
# This module will scale the Hyperparameter range and sample using Poisson Disc
# Sampling.
###############################################################################

import numpy as np
from  poisson_disc import *
import TwoLayerNeuralNet as net
import math
import time
import plotting_utils as plt

"""
poisson_disc_sampler takes samples and converts them to the scale of hyperparameters.

Input:
- learning_rate_range: Search Range for Learning Rate.
- momentum_range: Search Range for momentum_range.
- r: The Threshold radius of samples. (Density of samples increases as r
  decreseas, so does time taken.)

Returns:
- hyperparameters: List of Tuples. Tuple Stucture -(learning_rate, momentum).
- cells: Number of samples returned.
"""
def poisson_disc_sampler(learning_rate_range, momentum_range,r):
    # Initializing length and width. Can Be increased fo changing number of samples.
    length = 20
    width = 20

    # Sample Count.
    size = r / sqrt(2)
    cells = (length/size)* (width/size)

    # Initilizing Poisson Disc Sampling object.
    grid = Grid(r, length, width)

    # Random Seed.
    rand = (random.uniform(0, length), random.uniform(0, width))
    data = grid.poisson(rand)
    # # Zero Seed.
    # data = grid.poisson((0,0))

    step_size_l = learning_rate_range[1] - learning_rate_range[0]
    step_size_w = momentum_range[1]-momentum_range[0]
    hyperparameters = []

    # Converting Poisson Samples to Hyperparameter Samples.
    for i in range(0,len(data)):
        sample = data[i]
        learning_rate, momentum = learning_rate_range[0] + step_size_l*sample[0] / length, \
        momentum_range[0] + step_size_w*sample[1] / width
        hyperparameters.append((learning_rate, momentum))

    # print ("hyperparameters: ",hyperparameters)

    # For sample count. Stored in Cells.
    size = r / sqrt(2)
    cells = (length//size)* (width//size)

    return hyperparameters, cells

"""
Function to tune the hyperparameters. The learning_rate_range is converted to
lof, since, it gives better result.

Inputs:
-learning_rate_range : Takes Learnign Rate range default(1e-5,1e-1).
-momentum_range : Takes Momentum Range(0.5,1).

Returns:
-accuracy : Return Dictionary of accuracies with keys being the hyperparameters
used.
-acg_time_taken : Return Avg. Time Taken to train the NeuralNet
"""
def poisson_disc_search_tune(learning_rate_range = (1e-5,1e-1), momentum_range = (0.5,1)\
,verbose = False, use_logspace = False):

    accuracy = {}
    time_dict = {}

    # Converting to Linear Range if use_logspace True.
    if (use_logspace):
        learning_rate_range = ( math.log10(learning_rate_range[1]) , math.log10(learning_rate_range[0]) )

    hyperparameters,sample_count = poisson_disc_sampler(learning_rate_range,momentum_range, 2)

    for hyperparameter in hyperparameters:
        tick = time.time()
        if (use_logspace):
            learning_rate, momentum = 10**hyperparameter[0] ,hyperparameter[1]
        else:
            learning_rate, momentum = hyperparameter[0] ,hyperparameter[1]

        if (verbose):
            print ("learning_rate, momentum: ", learning_rate, momentum)

        # The NeuralNet is Being Trained For 50 Parameters.
        # Epochs Can be considered as a hyperparameter as well.
        accuracy[(learning_rate, momentum)] = net.compute\
        (learning_rate = learning_rate, momentum = momentum, epochs = 50)

        tock = time.time()
        time_dict[hyperparameter] = tock - tick

    avg_time_taken = sum(time_dict.values()) / len(time_dict)

    if (verbose):
        print ("Accuracy Dictionary: " , accuracy)
        print ("Avg. Time Taken in Sec: ",  avg_time_taken )

    return accuracy, avg_time_taken

"""
Initializing poisson disc search. Performs a sparse followed by a dense search.
"""
def initilize_poisson_disc_search(plot = False, verbose = False, use_logspace = False):
    # Declaring Parameters for function.
    learning_rate_range= (1e-5,1e-2)
    momentum_range = (0.7,1)

    # Dense Search range offsets. In percent.
    dense_lr_ofst, dense_mom_ofst = 20, 10


################################################################################
################################ Sparse Search #################################
################################################################################

    # Tuning Network on the Hyperparameter range. This block Will be treated as
    # a sparse search.
    accuracy, avg_time_taken = poisson_disc_search_tune(learning_rate_range =\
    learning_rate_range ,momentum_range = momentum_range,verbose = verbose, use_logspace = use_logspace)

    # Calculating best Accuracy.
    best_lr, best_mom = max(accuracy, key=accuracy.get)
    best_accuracy = accuracy[(best_lr,best_mom)]

    # Obtaining keys for plotting.
    keys = accuracy.keys()

    print ("Best Hyperparameter and Accuracy found form Sparse Search:"\
    , best_lr, best_mom, best_accuracy)

    if (plot):
        plt.plot_heatmap(accuracy= accuracy, file_name= "figures/poisson_heatmap.png")

################################################################################
################################ Dense Search ##################################
################################################################################

    # Readjusting Hyperparameters for Denser Search.
    learning_rate_range = ( best_lr - (best_lr * dense_lr_ofst / 100 ),\
    best_lr + (best_lr *dense_lr_ofst/ 100) )
    momentum_range = (best_mom - (best_mom * dense_mom_ofst / 100),\
    best_mom + (best_mom * dense_mom_ofst / 100))

    # Tuning Network on the Hyperparameter range. This block Will be treated as
    # a dense search.
    accuracy, avg_time_taken = poisson_disc_search_tune(learning_rate_range =\
    learning_rate_range ,momentum_range = momentum_range,verbose = verbose, use_logspace = use_logspace)

    # Calculating best Accuracy.
    best_lr_new, best_mom_new = max(accuracy, key=accuracy.get)
    best_accuracy_new = accuracy[(best_lr_new,best_mom_new)]

    # Getting keys as it will be used for plotting later.
    keys += accuracy.keys()

    print ("Best Hyperparameter and Accuracy found form Dense Search:"\
    , best_lr_new, best_mom_new, best_accuracy_new)

    # Selecting the new Hyperparameters if they give better results.
    if best_accuracy_new > best_accuracy:
        best_accuracy = best_accuracy_new
        best_lr = best_lr_new
        best_mom = best_mom_new

    #Appending Output to poisson disc sample file and plotting the samples.
    if (use_logspace):
        f = open('outputs/Output_poisson_disc_search_logspace.txt','a+')
        if (plot):
            plt.plot_hyperparameters(keys, "figures/poisson_logspace.png")
    else :
        f = open('outputs/Output_poisson_disc_search_regularspace.txt','a+')
        if (plot):
            plt.plot_hyperparameters(keys, "figures/poisson_regularspace.png")

    f.write('\nbest_lr, best_mom, best_accuracy ' +str(best_lr)+' , '+ \
    str(best_mom) +' , '+ str(best_accuracy) )
    f.close()
