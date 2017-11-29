import data_utils as du
import numpy as np

X_train, y_train, X_val, y_val, X_test, y_test = du.get_CIFAR10_data()
print ("X_train shape, y_train shape:", X_train.shape, y_train.shape)
