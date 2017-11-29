from datetime import datetime
import time
from matplotlib import pyplot as plt
from math import sqrt, pi
import seaborn
import random
from poisson_disc import Grid

def plot (dict,x_lable,y_lable):

    lists = sorted(dict.items())
    x,y = zip(*lists)
    plt.xlabel(x_lable)
    plt.ylabel(y_lable)
    plt.plot(x,y)
    plt.show()


def plot_poisson(data):
    plt.scatter(*unzip(data))
    # plt.scatter(10,5,'g*')
    # plt.scatter(11,6)
    # plt.axvline(ymin=2/14, ymax=12/14, color='red')
    # plt.axvline(x=20, ymin=2/14, ymax=12/14, color='red')
    # plt.axhline(y=10, xmin=5/30, xmax=25/30, color='red')
    # plt.axhline(y=0, xmin=5/30, xmax=25/30, color='red')
    plt.show()

def unzip(items):
    return ([item[i] for item in items] for i in range(len(items[0])))

## keeping the length and width fixed.
def compute_different_r(r):
    tick = time.time()
    length = 20
    width = 10

    # count of samples
    size = r / sqrt(2)
    cells = (length//size)* (width//size)

    #initialize object
    grid = Grid(r, length, width)

    #Random Seed
    rand = (random.uniform(0, length), random.uniform(0, width))
    data = grid.poisson(rand)
    # plot_poisson(data)

    tock =  time.time()

    # #Center Seed
    # rand = (length // 2, width // 2)
    # data = grid.poisson(rand)
    # # plot_poisson(data)

    return tock - tick, tock, tick, cells

sample_count = {}
time_dict = {}
rs = {0.3,0.5,0.7,0.9,1,2,3,4}

for r in rs:
    time_dict[r],tock,tick, sample_count[r] = compute_different_r(r)

print (time_dict)
print (tick, tock)
plot(time_dict, "r", "Time taken in mili sec")

print (sample_count)
plot(sample_count, "r", "Samples")
