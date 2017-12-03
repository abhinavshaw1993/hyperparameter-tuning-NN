import hyperparameter_space_scaler as hs
import numpy as np
import TwoLayerNeuralNet as net
import math
import time

"""
Function to tune the hyperparameters. The learning_rate_range is converted to lof, since, it gives better result.

Inputs:
-learning_rate_range : Takes Learnign Rate range default(1e-5,1e-1).
-momentum_range : Takes Momentum Range(0.5,1).

Returns:
-accuracy : Return Dictionary of accuracies with keys being the hyperparameters used.
-acg_time_taken : Return Avg. Time Taken to train the NeuralNet
"""
def tune_hyperparameters(learning_rate_range = (1e-5,1e-1), momentum_range = (0.5,1),verbose = False):

    # Converting to Linear Range.
    learning_rate_range = ( math.log10(learning_rate_range[1]) , math.log10(learning_rate_range[0]) )
    hyperparameters,sample_count = hs.scaler(learning_rate_range,momentum_range, 2)
    accuracy = {}
    time_dict = {}

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
