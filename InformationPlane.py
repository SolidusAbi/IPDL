import torch
from torch import nn

class MutualInformation(nn.Module):
    '''
        param step: number of steps in order to reduce the number of possible sigma 
        values.

        param sigma_values: number of possible sigma values for optimizing process.
    '''
    def __init__(self, sigma_values=75, step=150):
        # TODO
        return

    def forward(self, x, y):
        # TODO

        mean_distance_x = torch.tensor([torch.dist(x[i-1], x[i]) for i in range(1, len(x))]).mean()
        mean_distance_y = torch.tensor([torch.dist(y[i-1], y[i]) for i in range(1, len(y))]).mean()
        
        return

    '''
         Tensor Based Radial Basis Function (RBF) Kernel
    '''
    def RBF(self, x, sigma):
        distance = torch.cdist(x, x)
        return torch.exp(-(distance**2)/(sigma**2))
    
    '''
        Kernel Aligment Loss Function.

        This function is used in order to obtain the optimal sigma parameter from
        RBF kernel.  
    '''
    def kernelAligmentLoss(self, x, y):
        return (torch.sum(x*y))/(torch.norm(x) * torch.norm(y))

    def optimizeSigmaValue(self, x):
        '''
            This function is used in orter to obtain the optimal kernel width for
            an l DNN layer
        '''
        # TODO
        # note:
        #  input kernel width : 8
        #  label kernel width : 0.1
        distance = tensor[torch.dist(x[i-1], x[i]) for i in range(1, len(x))].mean()
        sigma_values = torch.arange(distance*0.1, distance*10, (distance*10 - distance*0.1)/75)
        return