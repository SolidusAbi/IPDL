import torch
import numpy as np
from torch import nn
# from numpy import linalg as LA
from scipy import linalg


class TensorKernel:
    '''
        Tensor Based Radial Basis Function (RBF) Kernel

        @param x
        @param sigma
    '''
    @staticmethod
    def RBF(x, sigma):
        distance = torch.cdist(x, x)
        return torch.exp(-(distance**2)/(sigma**2)).cpu()


class MatrixBasedRenyisEntropy():
    @staticmethod
    def entropy(A : np.array):
        # w, _ = LA.eig(A)
        w = linalg.eigh(A, eigvals_only=True)
        epsilon = 1e-6
        w += epsilon
        return -np.sum(w * np.log2(w))

    @staticmethod
    def jointEntropy(Ax : np.array, Ay : np.array):
        aux = Ax*Ay
        return MatrixBasedRenyisEntropy.entropy(aux/np.trace(aux))

    @staticmethod
    def mutualInformation(Ax : np.array, Ay : np.array):
        entropy_Ax = MatrixBasedRenyisEntropy.entropy(Ax)
        entropy_Ay = MatrixBasedRenyisEntropy.entropy(Ay)
        joint_entropy_AxAy = MatrixBasedRenyisEntropy.jointEntropy(Ax, Ay)
        return (entropy_Ax + entropy_Ay - joint_entropy_AxAy)

    '''
        Generates the 'A' matrix based on RBF kernel

        @return 'A' matrix
    '''
    @staticmethod
    def tensorRBFMatrix(x, sigma):
        return TensorKernel.RBF(x, sigma) / len(x)


class TensorRBFMatrix(nn.Module):
    def __init__(self):
        super(TensorRBFMatrix, self).__init__()
        self.sigma = None

    '''
        @param x
        @param beta regularizer term to stabilize the optimal sigma value across the previous iteration
    '''
    def forward(self, x, sigma):
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
    def step(self, layer_output, Ky, n_sigmas=75):
        sigma_t = self.optimize(layer_output, Ky, n_sigmas)
        self.sigma = ( (self.beta*self.sigma) + ((1-self.beta)*sigma_t) ) if not(self.sigma is None) else sigma_t
        return self.getSigma()

    '''
        This function is used in orter to obtain the optimal kernel width for
        an T DNN layer

        @param layer_output
        @param n_sigmas: number of possible sigma values

        [Descripción del procedimiento]
    '''
    def optimize(self, layer_output, Ky, n_sigmas):
        distance = torch.cdist(layer_output, layer_output)
        mean_distance = distance[distance != 0].mean().detach().cpu()
        sigma_values = np.arange(mean_distance*0.1, mean_distance*10, (mean_distance*10 - mean_distance*0.1)/n_sigmas)

        Kt = list( map(lambda sigma: TensorKernel.RBF(layer_output, sigma).detach(), sigma_values.tolist()) )
        loss = np.array( list( map(lambda k: self.kernelAligmentLoss(k, Ky), Kt) ) )
        self.sigma_tmp.append(sigma_values[ np.argwhere(loss == loss.max()).item(0) ])
        return sigma_values[ np.argwhere(loss == loss.max()).item(0) ]

    '''
        Kernel Aligment Loss Function.

        This function is used in order to obtain the optimal sigma parameter from
        RBF kernel.  
    '''
    def kernelAligmentLoss(self, x, y):
        return (torch.sum(x*y))/(torch.norm(x) * torch.norm(y))
        


class InformationPlane(torch.nn.Module):
    '''
        @param input_kernel: preprocessed input kernel matrix
        @param input_kernel: preprocessed label kernel matrix
        @param sigma_values: number of possible sigma values for optimizing process.
        @param step: indicates the number of step for reducing the number of possible sigma values
    '''
    def __init__(self, mini_batch_size, beta=0.5):
        super(InformationPlane, self).__init__()

        self.mini_batch_size = mini_batch_size
        self.sigma_optimizer = RKHSMatrixOptimizer(beta)
        self.Ixt = []
        self.Ity = []

    '''
        @param beta regularizer term to stabilize the optimal sigma value across the previous iteration
        
        @return mutual information with label {I(X,T), I(T,Y)}
    '''
    def forward(self, x, input, label, beta=0.5):
        original_shape = x.shape
        x = x.flatten(1)

        Ixt = [] # Mutual Information with input I(X, T)
        Ity = [] # Mutual Information with label I(T, Y)

        # Dividir en minibatchs [x]
        # Utilizar el optimizador [x]
        # Obtener la matrix A con el valor de sigma optimizado [x]

        for idx in range(0, len(x), self.mini_batch_size):
            batch = x[idx:idx+self.mini_batch_size]
            input_batch = input[idx:idx+self.mini_batch_size].flatten(1)
            label_batch = label[idx:idx+self.mini_batch_size]
            label_kernel_matrix = TensorKernel.RBF(label_batch, 0.1)
            
            self.sigma_optimizer.step(batch, label_kernel_matrix, 50)            

            A = MatrixBasedRenyisEntropy.tensorRBFMatrix(batch, self.sigma_optimizer.getSigma()).detach()
            Ay = MatrixBasedRenyisEntropy.tensorRBFMatrix(label_batch, 0.1).detach()
            Ax = MatrixBasedRenyisEntropy.tensorRBFMatrix(input_batch, 8).detach()
            Ixt.append(MatrixBasedRenyisEntropy.mutualInformation(Ax, A))
            Ity.append(MatrixBasedRenyisEntropy.mutualInformation(A, Ay))

        self.Ixt.append(np.array(Ixt).mean())
        self.Ity.append(np.array(Ity).mean())

        x = x.reshape(original_shape)
        return x

    ''' 
        @return Mutual Information {I(X,T), I(T,Y)}
    '''
    def getMutualInformation(self):
        return self.Ixt, self.Ity
