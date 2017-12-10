import numpy as np
import math
import TwoLayerNeuralNet as net
import time

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

def grid_search_tuner(learning_rate_range = (1e-5,1e-1), momentum_range = (0.5,1),verbose = False):
    #Some Initialization
    accuracy = {}
    time_dict = {}

    #converting Learning rate range to linear space using log. Done this for
    learning_rate_range = ( math.log10(learning_rate_range[1]) , math.log10(learning_rate_range[0]) )

    #Using Grid sampler.
    hyperparameters,sample_count = get_grid_hyperparameters(learning_rate_range,momentum_range, 2)

    for hyperparameter in hyperparameters:
        tick = time.time()
        learning_rate, momentum = 10**hyperparameter[0] ,hyperparameter[1]

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

grid_search_tuner()
