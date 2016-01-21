
import math
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import linregress

mpg_data_file = "./data/auto-mpg.txt"


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
            x3 = float(p[6])  # model year
            d.append((x1, x2, x3, y))
    return d


def plot(x, y, xlabel='', ylabel='', show=False, fname=None):
    plt.plot(x, y, 'o')
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if fname is not None:
        plt.savefig(fname)
    if show:
        plt.show()


def plot_inputs(x1, x2, x3, y):
    plot(x3, y, 'model year', 'mpg', fname='./mpg_year.png')
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


def plot_error_surface(x1, x2, y, b=50, fname='./error_surface.png'):
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    #W1 = np.arange(-1, 1, 0.05)
    #W2 = np.arange(-1, 1, 0.05)
    W1 = np.arange(-4000, 4000, 500)
    W2 = np.arange(-4000, 4000, 500)
    X, Y = np.meshgrid(W1, W2)
    zs = np.array([mse2(x1, x2, y, w1, w2, b) for w1, w2 in zip(np.ravel(X), np.ravel(Y))])
    Z = zs.reshape(X.shape)
    plt.xlabel('theta1')
    plt.ylabel('theta2')
    ax.plot_surface(X, Y, Z)
    plt.savefig(fname)
    plt.show()
    plt.figure()
    CS = plt.contour(X, Y, Z)
    plt.clabel(CS, inline=1, fontsize=10)
    plt.show()


def print_sweeps(x1, y):
    for b in np.arange(40, 60, 5):
        for w in np.arange(-0.5, 0, .05):
            print("{:.4f}\t{}\t{:>5.2f}".format(w, b, mse(x1, y, w, b)))


def print_sweeps_2(x1, x2, y):
    best = 10e7
    params = None
    for b in np.arange(-100, 100, 10):
        for w1 in np.arange(-1, 1, .05):
            for w2 in np.arange(-1, 1, .05):
                err = mse2(x1, x2, y, w1, w2, b)
                print("{:.4f}\t{:.4f}\t{}\t{:>5.2f}".format(w1, w2, b, err))
                if err < best:
                    best = err
                    params = (w1, w2, b)
    print(params, best)


def normalize(data):
    minx = min(data)
    maxx = max(data)
    return [(x-minx)/(maxx-minx) for x in data]

def plot_logit():
    x = np.arange(-5, 5, .1)
    y = 1 / (1+np.e**-x)
    plt.plot(x,y)
    plt.show()


def plot_logs():
    x = np.linspace(0.0001, 0.9999, 100)
    y1 = [-math.log(v) for v in x]
    y2 = [-math.log(1-v) for v in x]
    plt.plot(x, y1, label="-log(x)")
    plt.plot(x, y2, label="-log(1-x)")
    plt.legend()
    plt.show()


def plot_fit(x, y, mg=-0.0075, bg=50):
    r = linregress(x,y)
    m = r.slope
    b = r.intercept
    l1 = [m*p + b for p in x]
    l2 = [mg*p + bg for p in x]
    plt.plot(x, y, 'o')
    plt.xlabel('vehicle weight (lbs)')
    plt.ylabel('mpg')
    plt.plot(x, l1, label='best fit')
    plt.plot(x, l2, label='initial guess')
    plt.legend()
    plt.show()



def plot_fit(x, y, mg=-0.0075, bg=50):
    l2 = [mg*p + bg for p in x]
    plt.plot(x, y, 'o')
    plt.xlabel('vehicle weight (lbs)')
    plt.ylabel('mpg')
    plt.plot(x, l2, label='initial guess')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    data = read_mpg_data()
    x1, x2, x3, y = zip(*data)
    #plot_inputs(x1, x2, x3, y)
    #print_sweeps(x1, y)
    #plot_error_curve(x1, y)
    #plot_error_surface(x1, x3, y, fname='error_surface.png')
    #print_sweeps_2(x1, x3, y)
    #nx1 = normalize(x1)
    #nx2 = normalize(x2)
    #nx3 = normalize(x3)
    #plot_error_surface(nx1, nx3, y, fname='error_surface_norm.png')
    #plot_logit()
    #plot_logs()
    plot_fit(x2, y)