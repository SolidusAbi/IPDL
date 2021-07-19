import math
import numpy as np
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
from torch import Tensor
from torch.nn.functional import one_hot
from torch.distributions.dirichlet import Dirichlet

from .InformationPlane import InformationPlane

def showMutualInformation(ip: InformationPlane, moving_average_n = 15, colors: list = []):
    '''
        colors: https://matplotlib.org/3.1.0/tutorials/colors/colormaps.html
    '''
    if not colors:
        # Including Sequentials Colormaps
        colors = ['Greys', 'Reds', 'Blues', 'Greens', 'Oranges',
            'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
            'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
            'BuGn', 'YlGn']

    Ixts, Itys = ip.getMutualInformation(moving_average_n=moving_average_n)

    with plt.style.context('seaborn'):
        fig = plt.figure(constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.05, right=0.84, wspace=0.05)
        gs2 = fig.add_gridspec(nrows=1, ncols=len(Ixts), left=0.85, right=0.95, wspace=0)
        f8_ax1 = fig.add_subplot(gs1[:, :])
        f8_ax1.set_xlabel("I($X$, $T$)")
        f8_ax1.set_ylabel("I($T$, $Y$)")

        for idx, Ixt in enumerate(Ixts):
            Ity = Itys[idx]
            cmap = plt.cm.get_cmap(colors[idx])
            
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

def show_ip(ip_df: pd.DataFrame, n=25, labels=[], moving_average_n=25):
    '''
        Create a Information Plane illustration

        Parameters
        ----------
        ip_df : pd.Dataframe, dataframe which contains the MI. This dataframe
            has a specific structure which is created from IPDL.InformationPlane class
        n: int, Number of samples to visualize, sampling generated by a log-scale
    '''
    colors = ['Greys', 'Reds', 'Blues', 'Greens', 'Oranges',
                'Purples', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd',
                'RdPu', 'BuPu', 'GnBu', 'PuBu', 'YlGnBu', 'PuBuGn',
                'BuGn', 'YlGn']

    with plt.style.context('seaborn'):
        fig = plt.figure(constrained_layout=False)
        gs1 = fig.add_gridspec(nrows=1, ncols=1, left=0.08, right=0.84, wspace=0.05)
        gs2 = fig.add_gridspec(nrows=1, ncols=len(ip_df.keys()[::2]), left=0.85, right=0.95, wspace=0)
        f8_ax1 = fig.add_subplot(gs1[:, :])
        f8_ax1.set_xlabel("I($X$, $T$)", fontsize=14)
        f8_ax1.set_ylabel("I($T$, $Y$)", fontsize=14)

        for idx, (layer, _) in enumerate(ip_df.columns[::2]):
            Ity = moving_average(ip_df[layer]['Ity'].to_numpy(), n=moving_average_n, padding_size=int(moving_average_n*0.2))
            Ixt = moving_average(ip_df[layer]['Ixt'].to_numpy(), n=moving_average_n, padding_size=int(moving_average_n*0.2))
            
            cmap = plt.cm.get_cmap(colors[idx])
            iterations = np.geomspace(1, len(Ity)-1, num=n, dtype=np.uint)

            color = np.array([cmap(iterations[int(len(iterations)*0.75)])])
            sc = f8_ax1.scatter(Ixt[iterations], Ity[iterations], c=iterations, vmin=0, vmax=iterations.max(), cmap=cmap, edgecolor=color)
            if not labels:
                f8_ax1.scatter([], [], c=color, label=layer)
            else:
                f8_ax1.scatter([], [], c=color, label=labels[idx])

            f8_ax2 = fig.add_subplot(gs2[0, idx])
            cb = fig.colorbar(sc, cax=f8_ax2, pad=0)
            cb.set_ticks([])

        f8_ax1.legend()
        cb.set_ticks([0, iterations.max()])
        f8_ax2.set_yticklabels(['0', ip_df[layer]['Ixt'].size])
        cb.set_label("Iterations", labelpad=-18)

    return fig

def show_aeip(ip_df: pd.DataFrame, ref_entropy: float, n=25, moving_average_n=25):
    '''
        Create a Information Plane for AutoEncoder, encoder a decoder path has a 
        specific axis. 
        
        Dataframe is generated using InformationPlane class

        @param n: Number of samples to visualize, sampling generated by a log-scale
    '''
    markers = "od^spP"
    cmap = mpl.cm.Purples

    n_layers = int(len(ip_df.keys())/2)
    btnck_idx = math.ceil((n_layers/2))
    encoder_idx = np.arange(1, btnck_idx+1)
    decoder_idx = np.arange(btnck_idx-1, 0, step=-1)

    labels_encoder = list(map(lambda x: '$E_%d$' % x if x != btnck_idx else '$Z$', encoder_idx))
    labels_decoder = list(map(lambda x: '$D_%d$' % x if x != btnck_idx else '$Z$', decoder_idx))
    labels = labels_encoder + labels_decoder
    markers = markers[:btnck_idx] + markers[:btnck_idx-1][::-1]

    with plt.style.context('seaborn'):
        fig = plt.figure(constrained_layout=False, figsize=(16,8))
        gs1 = fig.add_gridspec(nrows=10, ncols=2, left=0.05, right=0.84, wspace=0.05, hspace=10)

        # Encoder axis
        f8_ax1 = fig.add_subplot(gs1[0:9, 0])
        f8_ax1.set_title("Encoder", fontsize=18)
        f8_ax1.set_xlabel("I($X$, $T$)", fontsize=14)
        f8_ax1.set_ylabel("I($T$, $Y$)", fontsize=14)
        f8_ax1.plot([0, 1], [0, 1], transform=f8_ax1.transAxes, linestyle='dashed')

        # Decoder axis
        f8_ax2 = fig.add_subplot(gs1[0:9, 1])
        f8_ax2.set_title("Decoder", fontsize=18) 
        f8_ax2.set_xlabel("I($X$, $T$)", fontsize=14)
        f8_ax2.set_ylabel("I($T$, $Y$)", fontsize=14, rotation=270, labelpad=20)
        f8_ax2.yaxis.tick_right()
        f8_ax2.yaxis.set_label_position("right")
        f8_ax2.plot([0, 1], [0, 1], transform=f8_ax2.transAxes, linestyle='dashed')

        for idx, (layer, _) in enumerate(ip_df.columns[::2]):
            Ixt = moving_average(ip_df[layer, 'Ixt'].to_numpy(), n=moving_average_n, padding_size=int(moving_average_n*0.2))
            Ity = moving_average(ip_df[layer, 'Ity'].to_numpy(), n=moving_average_n, padding_size=int(moving_average_n*0.2))
            iterations = np.geomspace(1, len(Ity)-1, num=n, dtype=np.uint) #Parametrizar 'num'

            if idx < btnck_idx-1:
                axis = f8_ax1
            else:   
                if idx == btnck_idx-1:
                    f8_ax1.scatter(Ixt[iterations], Ity[iterations], c=iterations, vmin=0, vmax=iterations.max(), label=labels[idx], marker=markers[idx], cmap=cmap, edgecolors='black', s=100)            
                axis = f8_ax2

            axis.scatter(Ixt[iterations], Ity[iterations], c=iterations, vmin=0, vmax=iterations.max(), label=labels[idx], marker=markers[idx], cmap=cmap, edgecolors='black', s=100)
            
                
        f8_ax1.legend()
        f8_ax1.set_xlim(xmax=ref_entropy)
        f8_ax1.set_ylim(ymax=ref_entropy)
        
        f8_ax2.legend()
        f8_ax2.set_xlim(xmax=ref_entropy)
        f8_ax2.set_ylim(ymax=ref_entropy)

        f8_ax3 = fig.add_subplot(gs1[9, :])
        f8_ax3.set_title("Iterations")
        norm = mpl.colors.Normalize(vmin=0, vmax=len(ip_df[layer, 'Ixt'].to_numpy()))
        cb1 = mpl.colorbar.ColorbarBase(f8_ax3, cmap=cmap,
                                        norm=norm,
                                        orientation='horizontal')
        return fig


def one_hot_dirichlet(x: Tensor, num_classes=-1):
    labels = one_hot(x, num_classes=num_classes).float()
    labels = (labels * 10000) + 100
    distribution = Dirichlet(labels)
    return distribution.sample()

def moving_average(a, n=10, padding_size=0) :
    ret = np.cumsum(a, dtype=np.float) if padding_size == 0 else np.cumsum(np.insert(a, 0, np.zeros(padding_size)), dtype=np.float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def gen_log_space(limit: int, n: int) -> np.ndarray:
    '''
        code from: https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
    '''
    result = [1]
    if n>1:  # just a check to avoid ZeroDivisionError
        ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    while len(result)<n:
        next_value = result[-1]*ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1]+1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit)/result[-1]) ** (1.0/(n-len(result)))
    
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x)-1, result)), dtype=np.uint64)
    