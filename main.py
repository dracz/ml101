
import numpy as np

mpg_data_file = "./data/auto-mpg.txt"

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import LinearLocator, FormatStrFormatter


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
            y = float(p[0])   # mpg
            x1 = float(p[2])  # displacement
            x2 = float(p[4])  # weight
            d.append((x1, x2, y))
    return d


def plot(x, y, xlabel='', ylabel='', show=False, fname=None):
    plt.clf()
    plt.plot(x, y, 'o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if fname is not None:
        plt.savefig(fname)
    if show:
        plt.show()


def plot_inputs(x1, x2, y):
    plot(x2, y, 'vehicle weight (lbs)', 'mpg', fname='./mpg_weight.png')
    plot(x1, y, 'engine displacement (cu in)', 'mpg', fname='./mpg_displacement.png')


def mse(x, y, w, b):
    m = len(x)
    total_err = 0
    for i in range(m):
        h = x[i] * w + b
        total_err += (y[i] - h) ** 2
    return total_err / m


def mse2(x1, x2, y, w1, w2, b):
    m = len(x1)
    total_err = 0
    for i in range(m):
        h = (x1[i] * w1) + (x2[i] * w2) + b
        total_err += (y[i] - h) ** 2
    return total_err / m


def plot_error_curve(x, y, w_min = -.3, w_max=.08, b=50):
    wr = np.linspace(w_min, w_max, 100)
    errs = [mse(x, y, w, b) for w in wr]
    plt.clf()
    plt.plot(wr, errs)
    plt.xlabel("theta1")
    plt.ylabel("cost")
    plt.title("cost curve for theta1, with b= {}".format(b))
    plt.savefig("./cost_theta1.png")
    plt.show()


def plot_error_curve2(x, y, w_min = -.3, w_max=.08, b=50):
    wr = np.linspace(w_min, w_max, 100)
    errs = [mse(x, y, w, b) for w in wr]
    plt.clf()
    plt.plot(wr, errs)
    plt.xlabel("theta1")
    plt.ylabel("cost")
    plt.title("cost curve for theta1, with b= {}".format(b))
    plt.savefig("./cost_theta1.png")
    plt.show()


def plot_error_surface(x1, x2, y, b=50):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    W1 = np.arange(-100, 100, 1)
    W2 = np.arange(-100, 100, 1)
    X, Y = np.meshgrid(W1, W2)
    zs = np.array([mse2(x1, x2, y, w1, w2, b) for w1, w2 in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)

    ax.plot_surface(X, Y, Z)
    plt.show()


def print_sweeps(x1, y):
    for b in np.arange(40, 60, 5):
        for w in np.arange(-0.5, 0, .05):
            print("{:.4f}\t{}\t{:>5.2f}".format(w, b, mse(x1, y, w, b)))


def print_sweeps_2(x1, x2, y):
    best = 10e7
    params = None
    for b in [0]:  #np.arange(0, 1000, 100):
        for w1 in np.arange(-1, 1, .1):
            for w2 in np.arange(-1, 1, .1):
                err = mse2(x1, x2, y, w1, w2, b)
                print("{:.4f}\t{:.4f}\t{}\t{:>5.2f}".format(w1, w2, b, err))
                if err < best:
                    best = err
                    params = (w1, w2, b)
    print(params, best)



if __name__ == "__main__":
    data = read_mpg_data()
    x1, x2, y = zip(*data)
    #plot_inputs(x1, x2, y)
    #print_sweeps(x1, y)
    #plot_error_curve(x1, y)
    #plot_error_surface(x1, x2, y)
    print_sweeps_2(x1, x2, y)
