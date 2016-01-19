

mpg_data_file = "./data/auto-mpg.txt"

import numpy as np
from matplotlib import pyplot as plt


def read_mpg_data():
    """
    1. mpg:           continuous
    2. cylinders:     multi-valued discrete
    3. displacement:  continuous
    4. horsepower:    continuous
    5. weight:        continuous
    6. acceleration:  continuous
    7. model year:    multi-valued discrete
    8. origin:        multi-valued discrete
    9. car name:      string (unique for each instance)
    """
    d = []
    with open(mpg_data_file) as f:
        for line in f:
            p = line.strip().split()
            y = float(p[0])
            x = float(p[4])
            d.append((x, y))
    return d


def plot(data, xlabel='', ylabel='', show=False, fname="./mpg.png"):
    x = [p[0] for p in data]
    y = [p[1] for p in data]
    plt.plot(x, y, 'o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if fname is not None:
        plt.savefig(fname)
    if show:
        plt.show()


if __name__ == "__main__":
    data = read_mpg_data()
    plot(data, 'vehicle weight (lbs)', 'mpg')

