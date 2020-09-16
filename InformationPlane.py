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
        return torch.exp(-(distance**2)/(sigma**2)).cpu()


class MatrixBasedRenyisEntropy():
    @staticmethod
    def entropy(A : np.array):
        w, _ = LA.eig(A)
        epsilon = 1e-10
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
    def __init__(self, label_kernel_matrix, beta=0.5):
        if not(0 <= beta <= 1):
            raise Exception('beta must be in the range [0, 1]')

        self.beta = beta
        self.Ky = label_kernel_matrix
        self.sigma = None

    def getSigma(self):
        return self.sigma

    '''
        @param The output of a specific layer
        @param label_kernel_matrix
        @param n_sigmas
    '''
    def step(self, layer_output, n_sigmas=75):
        sigma_t = optimize(layer_output, n_sigmas)
        self.sigma = ( (beta*self.sigma) + ((1-beta)*sigma_t) ) if not(self.sigma is None) else sigma_t
        return self.getSigma()

    '''
        This function is used in orter to obtain the optimal kernel width for
        an T DNN layer

        @param layer_output
        @param n_sigmas: number of possible sigma values

        [Descripción del procedimiento]
    '''
    def optimize(self, layer_output, n_sigmas):
        distance = torch.cdist(layer_output, layer_output)
        mean_distance = distance[distance != 0].mean()
        sigma_values = np.arange(mean_distance*0.1, mean_distance*10, (mean_distance*10 - mean_distance*0.1)/n_sigmas)

        Kt = list( map(lambda sigma: TensorKernel.RBF(layer_output, sigma), sigma_values.tolist()) )
        loss = np.array( list( map(lambda k: self.kernelAligmentLoss(k, self.Ky), Kt) ) )
        return sigma_values[ np.argwhere(loss == loss.max()).item(0) ]

    '''
        Kernel Aligment Loss Function.

        This function is used in order to obtain the optimal sigma parameter from
        RBF kernel.  
    '''
    def kernelAligmentLoss(self, x, y):
        return (torch.sum(x*y))/(torch.norm(x) * torch.norm(y))
        


class InformationPlane(nn.Module):
    '''
        @param input_kernel: preprocessed input kernel matrix
        @param input_kernel: preprocessed label kernel matrix
        @param sigma_values: number of possible sigma values for optimizing process.
        @param step: indicates the number of step for reducing the number of possible sigma values
    '''
    def __init__(self, mini_batch_size, input_kernel, label_kernel_matrix):
        super(InformationPlane, self).__init__()
        self.input_kernel = input_kernel
        self.label_kernel = label_kernel_matrix

        self.mini_batch_size = mini_batch_size
        self.sigma_optimizer = RKHSMatrixOptimizer(label_kernel_matrix, beta=0.5)

        self.Ixt = []
        self.Ity = []

    '''
        @param beta regularizer term to stabilize the optimal sigma value across the previous iteration
        
        @return mutual information with label {I(X,T), I(T,Y)}
    '''
    def forward(self, x, beta=0.5):
        original_shape = x.shape
        x = x.flatten(1)

        print(len(x))
        for idx in range(0, len(x), step=self.mini_batch_size):
            batch = x[idx:idx+self.mini_batch_size]
            A = self.tensorRBFMatrix(batch, 0.2)
            print(A)
        # Dividir en minibatchs
        # Utilizar el optimizador
        # Obtener la matrix A con el valor de sigma optimizado (self.tensorRBFMatrix(x, sigma))

        x = x.reshape(original_shape)
        return x

    ''' 
        @return Mutual Information {I(X,T), I(T,Y)}
    '''
    def getMutualInformation(self):
        return self.Ixt, self.Ity

    '''
        Generates the 'A' matrix based on RBF kernel

        @return 'A' matrix
    '''
    def tensorRBFMatrix(self, x, sigma):
        return TensorKernel.RBF(x, sigma) / len(x)

   
    # '''
    #     Kernel Aligment Loss Function.

    #     This function is used in order to obtain the optimal sigma parameter from
    #     RBF kernel.  
    # '''
    # def kernelAligmentLoss(self, x, y):
    #     return (torch.sum(x*y))/(torch.norm(x) * torch.norm(y))

    # '''
    #     This function is used in orter to obtain the optimal kernel width for
    #     an L DNN layer

    #     @param x
    #     @param n_sigmas: number of possible sigma values

    #     [Descripción del procedimiento]
    # '''
    # def optimizeSigmaValue(self, x, n_sigmas=75):
    #     distance = torch.cdist(x, x)
    #     mean_distance = distance[distance != 0].mean()
    #     sigma_values = np.arange(mean_distance*0.1, mean_distance*10, (mean_distance*10 - mean_distance*0.1)/n_sigmas)
        
    #     rbf_result = list( map(lambda sigma: TensorKernel.RBF(x, sigma), sigma_values.tolist()) )
    #     loss = np.array( list( map(lambda x: self.kernelAligmentLoss(x, self.label_kernel), rbf_result) ) )
    #     optimal_sigma_value = sigma_values[ np.argwhere(loss == loss.max()).item() ]

    #     return optimal_sigma_value
