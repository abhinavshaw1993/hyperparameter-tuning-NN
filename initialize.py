import tuner

"""
Main Function to start the procedure.
"""
def main():
    # Declaring Parameters for tuner function.
    learning_rate_range= (1e-4,1e-2)
    momentum_range = (0.7,1)
    verbose = True
    # Dense Search range offsets. In percent.
    dense_lr_ofst, dense_mom_ofst = 50, 10

    # Tuning Network on the Hyperparameter range. This block Will be treated as
    # a sparse search.
    accuracy, avg_time_taken = tuner.tune_hyperparameters(learning_rate_range =\
    learning_rate_range ,momentum_range = momentum_range,verbose = True)

    #Calculating best Accuracy.
    best_lr, best_mom = max(accuracy, key=accuracy.get)
    best_accuracy = accuracy[best_hyperparameter]

    # best_lr, best_mom = (0.0006694521324073282, 0.7933857494299539)
    # best_accuracy = 34.5
    # dense_lr_ofst, dense_mom_ofst = 30, 10

    print ("Best Hyperparameter and Accuracu found form Sparse Search:"\
    , best_lr, best_mom, best_accuracy)

    # Readjusting Hyperparameters for Denser Search.
    learning_rate_range = ( best_lr - (best_lr * dense_lr_ofst / 100 ),\
    best_lr + (best_lr *dense_lr_ofst/ 100) )
    momentum_range = (best_mom - (best_mom * dense_mom_ofst / 100),\
    best_mom + (best_mom * dense_mom_ofst / 100))
    # print ("learning_rate_range, momentum_range: ",learning_rate_range , momentum_range)

    # Tuning Network on the Hyperparameter range. This block Will be treated as
    # a sparse search.
    accuracy, avg_time_taken = tuner.tune_hyperparameters(learning_rate_range =\
    learning_rate_range ,momentum_range = momentum_range,verbose = True)

    #Calculating best Accuracy.
    best_lr, best_mom = max(accuracy, key=accuracy.get)
    best_accuracy = accuracy[best_hyperparameter]

    print ("Best Hyperparameter and Accuracu found form Dense Search:"\
    , best_lr, best_mom, best_accuracy)
