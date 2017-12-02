import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import data_utils as du

"""
Function to plot Graph.
"""
def plot_graph(plot_dict):
    # lists = sorted(plot_dict.items())
    lists = plot_dict.items()
    x,y = zip(*lists)
    plt.plot(x,y)
    plt.show()

class TwoLayerNeuralNet (torch.nn.Module):
    # Declaring Variables.
    X_train, y_train = None, None
    X_val, y_val = None, None
    X_test, y_test = None, None

    def __init__(self,D_in,H,D_out):
        super(TwoLayerNeuralNet,self).__init__()
        self.linear1 = nn.Linear(D_in,H)
        self.linear2 = nn.Linear(H,D_out)

    """
    Forward Pass Definition.
    """
    def forward(self, x):
        h_relu = self.linear1(x).clamp(min = 0)
        y_pred = self.linear2(h_relu)
        return y_pred

    # def CIFAR_data_getter(self,num_training,num_validation,num_test):
    #     self.X_train, self.y_train, self.X_val, self.y_val, self.X_test, self.y_test\
    #     = du.get_CIFAR10_data(num_training=4000, num_validation=1000, num_test=200)

"""
Compute Function to Train the Network.

The notwork is trained on a subset of CIFAR-10 images, this nu,ber is
configurable in the code below. Our Neural Network is a fully-connected
two-layer NN.

Inputs:
- epochs: The number of epochs that you want to train the network for.
  (Recommended Value: 100)
- learning_rate (Hyperparameter): Learning Rate of the Network.
  This will be decided by the Poisson Disc Sampler.
- momentun (Hyperparameter): Momentum for optimizer.
  This will be decided by the Poisson Disc Sampler.

Returns:
- accuracy: The accuracy of the network.
"""
def compute( learning_rate, momentum, epochs = 50):
    #loading with CIFAR Data.
    X_train, y_train, X_val, y_val, X_test, y_test =\
    du.get_CIFAR10_data(num_training=10000, num_validation=10, num_test=1000)

    # Initializing NeuralNet Dimensions.
    N,D_in = X_train.shape
    H, D_out = 50,10
    dtype = torch.FloatTensor

    # Converting NumPy Array to torch tensor.
    X_train = torch.from_numpy(X_train).float()
    y_train = torch.LongTensor(y_train)

    #Wrapping with Variable.
    x = Variable(X_train)
    y = Variable(y_train,requires_grad = False)

    # Initializing TwoLayerNeuralNet.
    model = TwoLayerNeuralNet(D_in, H, D_out)

    # Loss Function and Optimizer.
    # critirion = torch.nn.MSELoss(size_average=False)
    critirion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(),lr = learning_rate, momentum=momentum)
    loss_dict = {}

    for i in range(epochs):
        y_pred = model(x)
        loss = critirion(y_pred,y)
        loss_dict[i] = loss.data[0]
        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Converting Test NumPy Array to torch tensor.
    X_test = torch.from_numpy(X_test).float()
    y_test = torch.from_numpy(y_test).float()
    X_test = Variable(X_test)

    # Testing the accuracy.Max returns both values and indices.
    y_test_pred = model(X_test)
    _ , y_test_pred = torch.max(y_test_pred.data,1)

    # Calculating total and corrct. Conversion required since target y_test_pred is a long tensor.
    total_labels = y_test.size(0)
    correct = (y_test_pred.float() == y_test).sum()
    accuracy = (100.0 * correct/total_labels, correct)

    # Checking Accuracy and plotting graph.
    # print ( "Accuracy : ", 100.0 * correct/total_labels, correct)
    # plot_graph(loss_dict)

    # Accuracy + Correct Count.
    return accuracy

# print (compute( 5e-5, 0.85, 50))
# print (compute( 1e-5, 0.9, 50))
