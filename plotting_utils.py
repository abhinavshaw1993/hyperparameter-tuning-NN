import matplotlib.pyplot as plt
import numpy as np
import TwoLayerNeuralNet as nn

def plot_hyperparameters(keys, file_name):
    y_axis = []
    x_axis = []
    plt.clf()

    for i in range(len(keys)):
        y_axis.append(keys[i][0])
        x_axis.append(keys[i][1])

    if "grid" in file_name:
        markersize = 0.75
    else:
        markersize = 1.5

    plt.plot(x_axis, y_axis, 'ro', markersize = markersize)
    plt.title("Learning Rate vs momentum")
    plt.xlabel ('Momentum')
    plt.ylabel ('Learning Rate')
    plt.grid(True)
    plt.savefig(file_name)

def plot_heatmap (accuracy, file_name):
    print ("Implement here")

    # Extracting the Accuracy and hyperparameters.
    keys = np.array(accuracy.keys())
    z = np.array(accuracy.values())
    z = z[:,0]

    # Plotting the scatterplot.
    plt.scatter(keys[:,1], keys[:,0], c=z, s=2, cmap = 'hot')
    plt.colorbar()
    plt.xlabel("Momentum")
    plt.ylabel("Learning Rate")
    plt.savefig(file_name)

# def plot_contour ():
#     # Function to Generate a heat map.
