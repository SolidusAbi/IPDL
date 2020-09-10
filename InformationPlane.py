import torch
import numpy as np
from torch import nn


class TensorKernel:
    '''
        Tensor Based Radial Basis Function (RBF) Kernel

        @param x
        @param sigma
    '''
    def RBF(x, sigma):
        distance = torch.cdist(x, x)
        return torch.exp(-(distance**2)/(sigma**2))

class MutualInformation(nn.Module):
    '''
        param step: number of steps in order to reduce the number of possible sigma 
        values.

        @param input_kernel: preprocessed input kernel matrix
        @param input_kernel: preprocessed label kernel matrix
        @param sigma_values: number of possible sigma values for optimizing process.
        @param step: indicates the number of step for reducing the number of possible sigma values
    '''
    def __init__(self, input_kernel, label_kernel, sigma_values=75, step=150):
        self.input_kernel = input_kernel
        self.label_kernel = label_kernel
        self.sigma_prev = None

    def forward(self, x, y):
        # TODO
        # mean_distance_x = torch.tensor([torch.dist(x[i-1], x[i]) for i in range(1, len(x))]).mean()
        # mean_distance_y = torch.tensor([torch.dist(y[i-1], y[i]) for i in range(1, len(y))]).mean()
        return
   
    '''
        Kernel Aligment Loss Function.

        This function is used in order to obtain the optimal sigma parameter from
        RBF kernel.  
    '''
    def kernelAligmentLoss(self, x, y):
        return (torch.sum(x*y))/(torch.norm(x) * torch.norm(y))

    '''
        This function is used in orter to obtain the optimal kernel width for
        an L DNN layer

        @param x
        @param n_sigmas: number of possible sigma values

        [Descripción del procedimiento]
    '''
    def optimizeSigmaValue(self, x, n_sigmas=75):
        distance = torch.cdist(x, x)
        mean_distance = distance[distance != 0].mean()
        sigma_values = np.arange(mean_distance*0.1, mean_distance*10, (mean_distance*10 - mean_distance*0.1)/n_sigmas)
        
        rbf_result = list( map(lambda sigma: TensorKernel.RBF(x, sigma), sigma_values.tolist()) )
        loss = np.array( list( map(lambda x: self.kernelAligmentLoss(x, self.label_kernel), rbf_result) ) )
        best_sigma_value = sigma_values[ np.argwhere(loss == loss.max()).item() ]

        return best_sigma_value
