
import math 
import torch
from torch import nn
from functools import reduce
from .AligmentOptimizer import MatrixOptimizer

class SivermanOptimizer(MatrixOptimizer):
    '''
        A simplified approach of Silverman’s rule of thumb for Gaussian kernels.
        This class is based on the approach proposed in "On the Information
        Plane of Autoencoders". 

        Eq:
        "\\sigma = \\gamma\\sqrt{d}N^{ ({-1}/(4+d)) }"
    '''

    def __init__(self, model: nn.Module, gamma: float = 0.8, normalize_dim = True):
        '''
        @param gamma
        @param normalize_dim: Check if the dimensional normalization in the equation 
            is applied
        '''
        super(SivermanOptimizer, self).__init__(model)

        self.gamma = gamma
        self.normalize = normalize_dim

    def step(self):
        self.__optimize()

    def __optimize(self) -> None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        for matrix_estimator in self.matrix_estimators:
            x = (matrix_estimator.x).to(device)

        n = x.size(0)
        d = x.size(1) if len(x.shape) == 2 else reduce(lambda x, y: x*y, x.shape[1:])
        
        sigma = self.gamma * (n ** (-1 / (4 + d)))
        if self.normalize:
            sigma = sigma * math.sqrt(d)

        matrix_estimator.set_sigma(sigma)

        return x