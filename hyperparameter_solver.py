import hyperparameter_space_scaler as hs
import numpy as np
import TwoLayerNeuralNet as net
import math

def get_best_hyperparameter(learning_rate_range = (1e-8,1e-1), momentum_range = (0.5,1)):

    print (learning_rate_range)
    # learning_rate_range = (6,1)
    # momentum_range = (0.5,1)
    learning_rate_range = ( math.log10(learning_rate_range[1]) , math.log10(learning_rate_range[0]) )
    print ("learning_rate_range: ",learning_rate_range)

    # # Converting to Linear Range.
    # learning_rate_range = ( )

    hyperparameters,sample_count = hs.scaler(learning_rate_range,momentum_range, 2)
    accuracy_dict = {}

    # print ("hyperparameters from solver, Count: ",hyperparameters, sample_count)

    # for hyperparameter in hyperparameters:
    #     learning_rate, momentum = hyperparameter
        # print ("hyperparameter: ", hyperparameter)
        # accuracy_dict[hyperparameter] = net.compute(learning_rate = learning_rate, momentum = momentum, epochs = 100)

    # print ("Accuracy Dictionary: " , accuracy_dict)

get_best_hyperparameter()
