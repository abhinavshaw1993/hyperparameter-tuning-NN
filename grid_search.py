###############################################################################
# This module will scale the Hyperparameter range and sample using Grid Search.
###############################################################################

import numpy as np
import math
import TwoLayerNeuralNet as net
import time
import plotting_utils as plt

"""
Function to get hyperparameters from grid search.

Inputs:
-learning_rate_range : Takes Learning Rate range.
-momentum_range : Takes Momentum Range.

Returns:
-hyperparameters : list of hyperparameters that are calcualted from grid search.
-sample_count : Np. of samples from grid search.
"""
def get_grid_hyperparameters(learning_rate_range,momentum_range, r):

    # Initializing length and width. Can Be increased fo changing number of samples.
    length = 20
    width = 20

    #Seed
    seed = (0,0)

    #samples.
    hyperparameters = []

    #step size learning rate and momentum
    step_size_l = learning_rate_range[1] - learning_rate_range[0]
    step_size_w = momentum_range[1]-momentum_range[0]

    #Sample count. samples returned for a grid search.
    sample_count =( length / r + 1) * (width / r + 1)

    for i in range(0,(length/ r) + 1):
        for j in range(0,(width/ r) + 1):
            lr = learning_rate_range[0] + step_size_l * i / length
            mom = momentum_range[0] + step_size_w * j / width
            hyperparameters.append ((lr,mom))

    # print ("Grid Search, Sample Count: ", hyperparameters , len(hyperparameters))
    return hyperparameters, sample_count

"""
Function that gets hyperparameters from grid search and train neural network on
all of the hyperparameters.

Inputs:
-learning_rate_range : Takes Learning Rate range.
-momentum_range : Takes Momentum Range.
-vebose : will print if True.

Returns:
-accuracy : list of accuracy from every pair of learning rate and momentum.
-avg_time_taken : avg. time taken to train using ever set of hyperparameters.
"""
def grid_search_tuner(learning_rate_range = (1e-5,1e-1), momentum_range = (0.5,1)\
,verbose = False, use_logspace = False):

    #Some Initialization
    accuracy = {}
    time_dict = {}

    #converting Learning rate range to linear space using log. Done this for
    if (use_logspace):
        learning_rate_range = ( math.log10(learning_rate_range[1]) , math.log10(learning_rate_range[0]) )

    #Using Grid sampler.
    hyperparameters,sample_count = get_grid_hyperparameters(learning_rate_range,momentum_range, 2)

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
Function for initializing grid search. It des a sparse search and the does a
dense search. This Way we save on computation well.

"""
def initilize_grid_search(plot = False, verbose = False, use_logspace = False):

    # Declaring Parameters for tuner function.
    learning_rate_range= (1e-5,1e-2)
    momentum_range = (0.7,1)

    # Dense Search range offsets. In percent.
    dense_lr_ofst, dense_mom_ofst = 20, 10

################################################################################
################################ Sparse Search #################################
################################################################################

    # Tuning Network on the Hyperparameter range. This block Will be treated as
    # a sparse search.
    accuracy, avg_time_taken = grid_search_tuner(learning_rate_range =\
    learning_rate_range ,momentum_range = momentum_range,verbose = verbose, use_logspace = use_logspace)

    #Calculating best Accuracy.
    best_lr, best_mom = max(accuracy, key=accuracy.get)
    best_accuracy = accuracy[(best_lr,best_mom)]

    #  Obtaining Keys for plotting
    keys = accuracy.keys()

    print ("Best Hyperparameter and Accuracy found form Sparse Search of Grid Search:"\
    , best_lr, best_mom, best_accuracy)

    if (plot):
        plt.plot_heatmap(accuracy= accuracy, file_name= "figures/grid_heatmap.png")

################################################################################
################################ Dense Search ##################################
################################################################################

    # Readjusting Hyperparameters for Denser Search.
    learning_rate_range = ( best_lr - (best_lr * dense_lr_ofst / 100 ),\
    best_lr + (best_lr *dense_lr_ofst/ 100) )
    momentum_range = (best_mom - (best_mom * dense_mom_ofst / 100),\
    best_mom + (best_mom * dense_mom_ofst / 100))
    # print ("learning_rate_range, momentum_range: ",learning_rate_range , momentum_range)

    # Tuning Network on the Hyperparameter range. This block Will be treated as
    # a sparse search.
    accuracy, avg_time_taken = grid_search_tuner(learning_rate_range =\
    learning_rate_range ,momentum_range = momentum_range,verbose = verbose, use_logspace = use_logspace)


    #Calculating best Accuracy.
    best_lr_new, best_mom_new = max(accuracy, key=accuracy.get)
    best_accuracy_new = accuracy[(best_lr,best_mom)]

    keys += accuracy.keys()

    print ("Best Hyperparameter and Accuracy found form Dense Search Grid Search: "\
    , best_lr_new, best_mom_new, best_accuracy_new)

    # Selecting the new Hyperparameters if they give better results.
    if best_accuracy_new > best_accuracy:
        best_accuracy = best_accuracy_new
        best_lr = best_lr_new
        best_mom = best_mom_new

    # Writing in file and plotting if plot is True.
    if (use_logspace):
        f = open('outputs/Output_grid_search_logspace.txt','a+')
        if (plot):
            plt.plot_hyperparameters(keys, "figures/grid_logspace.png")
    else :
        f = open('outputs/Output_grid_search_regularspace.txt','a+')
        if (plot):
            plt.plot_hyperparameters(keys, "figures/grid_regularspace.png")

    #Writing file in append mode Saving data for calculation.
    f.write('\nbest_lr, best_mom, best_accuracy ' +str(best_lr)+' , '+ str(best_mom) +' , '+ str(best_accuracy) )
    f.close()
