import numpy as np
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.functional import one_hot
from torch.distributions.dirichlet import Dirichlet

'''
    Colors: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    Ex:
        colors = ['Reds', 'Blues', 'binary', 'Greens', 'Oranges']   
'''
def showMutualInformation(ip_layers: list, colors: list):
    if len(ip_layers) > len(colors):
        raise ValueError("Each layer needs a color")

    with plt.style.context('seaborn'):
        fig = plt.figure(constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.84, wspace=0.05)
        gs2 = fig.add_gridspec(nrows=1, ncols=5, left=0.85, right=0.95, wspace=0)
        f8_ax1 = fig.add_subplot(gs1[:, :])
        f8_ax1.set_xlabel("I(X, T)")
        f8_ax1.set_ylabel("I(T, Y)")

        for idx, ip in enumerate(ip_layers):
            cmap = plt.cm.get_cmap(colors[idx])
            Ixt, Ity = ip.getMutualInformation()
            Ixt = moving_average(Ixt)
            Ity = moving_average(Ity)
            iterations = np.arange(len(Ixt))
            color = np.array([cmap(iterations[-1])])
            sc = f8_ax1.scatter(Ixt, Ity, c=iterations, vmin=0, vmax=iterations.max(), cmap=cmap, edgecolor=color)
            f8_ax1.scatter([], [], c=color, label="Layer {}".format(idx))

            f8_ax2 = fig.add_subplot(gs2[0, idx])
            cb = fig.colorbar(sc, cax=f8_ax2, pad=0)
            cb.set_ticks([])

        f8_ax1.legend()
        cb.set_ticks([0, iterations.max()])
        cb.set_label("Iterations", labelpad=-18)

        plt.show()

def one_hot_dirichlet(x: Tensor, num_classes=-1):
    labels = one_hot(x, num_classes=num_classes).float()
    labels = (labels * 10000) + 100
    distribution = Dirichlet(labels)
    return distribution.sample()

def moving_average(a, n=10) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n