import torch
import numpy as np
from torch import nn
from torch import Tensor


class TensorKernel:
    '''
        Tensor Based Radial Basis Function (RBF) Kernel

        @param x
        @param sigma
    '''
    @staticmethod
    def RBF(x: Tensor, sigma: float) -> Tensor:
        distance = torch.cdist(x, x)
        return torch.exp(-distance**2 / (sigma**2) )


class MatrixBasedRenyisEntropy():
    @staticmethod
    def entropy(A: Tensor) -> float:
        eigv = torch.symeig(A)[0].abs()
        epsilon = 1e-12
        eigv += epsilon 
        return -torch.sum(eigv*(torch.log2(eigv)))

    @staticmethod
    def jointEntropy(*args: Tensor) -> float:
        for idx, val in enumerate(args):
            if idx==0:
                A = val.clone()
            else:
                A *= val
        
        A /= A.trace()
        return MatrixBasedRenyisEntropy.entropy(A)

    @staticmethod
    def mutualInformation(Kx: Tensor, Ky: Tensor) -> float:
        entropy_Ax = MatrixBasedRenyisEntropy.entropy(Kx)
        entropy_Ay = MatrixBasedRenyisEntropy.entropy(Ky)
        joint_entropy = MatrixBasedRenyisEntropy.jointEntropy(Kx, Ky)
        return (entropy_Ax + entropy_Ay - joint_entropy)


    '''
        Generates the 'A' matrix based on RBF kernel

        @return 'A' matrix
    '''
    @staticmethod
    def tensorRBFMatrix(x: Tensor, sigma: float) -> Tensor:
        return TensorKernel.RBF(x, sigma) / len(x)


class RKHSMatrixOptimizer():
    def __init__(self, beta=0.5):
        if not(0 <= beta <= 1):
            raise Exception('beta must be in the range [0, 1]')

        self.beta = beta
        self.sigma = None
        self.sigma_tmp = [] #Just for saving sigma values

    # Temporal, just for testing
    def getSigmaValues(self):
        return self.sigma_tmp

    def getSigma(self):
        return self.sigma

    '''
        @param The output of a specific layer
        @param label_kernel_matrix
        @param n_sigmas
    '''
    def step(self, layer_output: Tensor, Ky: Tensor, sigma_values: list) -> float:
        sigma_t = self.optimize(layer_output, Ky, sigma_values)
        self.sigma = ( (self.beta*sigma_t) + ((1-self.beta)*self.sigma) ) if not(self.sigma is None) else sigma_t
        return self.getSigma()

    '''
        This function is used in orter to obtain the optimal kernel width for
        an T DNN layer

        @param layer_output
        @param n_sigmas: number of possible sigma values

        [Descripción del procedimiento]
    '''
    def optimize(self, x: Tensor, Ky: Tensor, sigma_values: list) -> float:
        Kt = list( map(lambda sigma: TensorKernel.RBF(x, sigma).detach(), sigma_values) )
        loss = np.array( list( map(lambda k: self.kernelAligmentLoss(k, Ky), Kt) ) )

        self.sigma_tmp.append(sigma_values[ np.argwhere(loss == loss.max()).item(0) ])
        return self.sigma_tmp[-1]

    '''
        Kernel Aligment Loss Function.

        This function is used in order to obtain the optimal sigma parameter from
        RBF kernel.  
    '''
    def kernelAligmentLoss(self, x: Tensor, y: Tensor) -> float:
        return (torch.sum(x*y)/(torch.norm(x) * torch.norm(y))).item()
        


class InformationPlane(torch.nn.Module):
    '''
        @param input_kernel: preprocessed input kernel matrix
        @param input_kernel: preprocessed label kernel matrix
        @param sigma_values: number of possible sigma values for optimizing process.
        @param step: indicates the number of step for reducing the number of possible sigma values
    '''
    def __init__(self, beta=0.5, n_sigmas=75):
        super(InformationPlane, self).__init__()

        self.sigma_optimizer = RKHSMatrixOptimizer(beta)
        self.Ixt = []
        self.Ity = []

        self.input_batch = None
        self.label_batch = None
        self.n_sigmas=n_sigmas

    def setNumberOfSigma(self, n_sigmas):
        self.n_sigmas = n_sigmas

    '''
        It's necessary to update the X and Y, input and label, in each iteration.

        @param input: batch with the original input
        @param label: label of the data
    '''
    def setInputLabel(self, inputs: Tensor, labels: Tensor):
        self.input_batch = inputs.flatten(1)
        self.label_batch = labels

    '''
        @return mutual information with label {I(X,T), I(T,Y)}
    '''
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            return x
        
        original_shape = x.shape
        x = x.flatten(1)
        
        sigma_values = self.getPossibleSigmaValues(x)
        Ky = TensorKernel.RBF(self.label_batch, 0.1)
        best_sigma = self.sigma_optimizer.step(x, Ky, sigma_values)

        A = MatrixBasedRenyisEntropy.tensorRBFMatrix(x, best_sigma).detach()
        Ay = MatrixBasedRenyisEntropy.tensorRBFMatrix(self.label_batch, 0.1).detach()
        Ax = MatrixBasedRenyisEntropy.tensorRBFMatrix(self.input_batch, 8).detach()

        self.Ixt.append(MatrixBasedRenyisEntropy.mutualInformation(Ax, A))
        self.Ity.append(MatrixBasedRenyisEntropy.mutualInformation(A, Ay))

        x = x.reshape(original_shape)
        return x

    '''
        Defines an array which contains the possible sigma values in 1-D array. The number of possible
        sigma values can be modified using the function setNumberOfSigma().

        @param x: Batch tensor
    '''
    def getPossibleSigmaValues(self, x: Tensor) -> list:
        distance = torch.cdist(x, x)
        mean_distance = distance[~torch.eye(len(distance), dtype=bool)].mean().item()
        return torch.linspace(0.1, mean_distance*10, self.n_sigmas).tolist()

    def moving_average(x: Tensor, n=10) -> Tensor :
        ret = torch.cumsum(x, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    ''' 
        @return Mutual Information {I(X,T), I(T,Y)}
    '''
    def getMutualInformation(self):
        return self.Ixt, self.Ity
