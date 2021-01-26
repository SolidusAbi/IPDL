import torch 
from torch import Tensor

class TensorKernel:    
    @staticmethod
    def RBF(x: Tensor, sigma: float) -> Tensor:
        '''
            Tensor Based Radial Basis Function (RBF) Kernel

            @param x
            @param sigma
        '''
        pairwise_difference = (torch.unsqueeze(x,1) - torch.unsqueeze(x,0))**2
        distance = torch.sum(pairwise_difference, dim=2)
        # distance = torch.cdist(x, x)
        return torch.exp(-distance / (2*(sigma**2)) )

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



    @staticmethod
    def tensorRBFMatrix(x: Tensor, sigma: float) -> Tensor:
        '''
            Generates the 'A' matrix based on RBF kernel

            @return 'A' matrix
        '''
        return TensorKernel.RBF(x, sigma) / len(x)
