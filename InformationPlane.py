import torch
import numpy as np
from torch import nn
from numpy import linalg as LA

class TensorKernel:
    '''
        Tensor Based Radial Basis Function (RBF) Kernel

        @param x
        @param sigma
    '''
    @staticmethod
    def RBF(x, sigma):
        distance = torch.cdist(x, x)
        return torch.exp(-(distance**2)/(sigma**2))

class MatrixBasedEntropy:
    @staticmethod
    def renyisEntropy(A : np.array):
        w, _ = LA.eig(A)
        epsilon = 1e-10
        w += epsilon
        return -np.sum(w * np.log2(w))

    @staticmethod
    def jointEntropy(A_x : np.array, A_y : np.array):
        aux = A_x*A_y
        return matrixRenyiEntropy(aux/np.trace(aux))

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
        self.sigma = None

    '''
        @param beta regularizer term to stabilize the optimal sigma value across the previous iteration
    '''
    def forward(self, x, beta=0.5):
        if not(0 <= beta <= 1):
            raise Exception('beta must be in the range [0, 1]')
        
        if not(self.sigma is None):
            self.sigma = (beta*self.sigma) + ((1-beta)*self.optimizeSigmaValue(x))
        else:
            self.sigma = self.optimizeSigmaValue(x)

        A = TensorKernel.RBF(x, self.sigma) / len(x)

        #TODO
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
        optimal_sigma_value = sigma_values[ np.argwhere(loss == loss.max()).item() ]

        return optimal_sigma_value
