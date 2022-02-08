import torch 
from torch import Tensor

# class TensorKernel:    
#     @staticmethod
#     def RBF(x: Tensor, sigma: float) -> Tensor:
#         '''
#             Tensor Based Radial Basis Function (RBF) Kernel

#             @param x
#             @param sigma
#         '''
#         pairwise_difference = (torch.unsqueeze(x,1) - torch.unsqueeze(x,0))**2
#         distance = torch.sum(pairwise_difference, dim=2)
#         # distance = torch.cdist(x, x)
#         return torch.exp(-distance / (2*(sigma**2)) )


# class MatrixBasedRenyisEntropy():
#     @staticmethod
#     def entropy(A: Tensor) -> float:
#         eigval, _ = torch.linalg.eigh(A)        
#         epsilon = 1e-8
#         eigval = eigval.abs() + epsilon 
#         return -torch.sum(eigval*(torch.log2(eigval)))

#     @staticmethod
#     def jointEntropy(*args: Tensor) -> float:
#         for idx, val in enumerate(args):
#             if idx==0:
#                 A = val.clone()
#             else:
#                 A *= val
        
#         A /= A.trace()
#         return MatrixBasedRenyisEntropy.entropy(A)

#     @staticmethod
#     def mutualInformation(Kx: Tensor, Ky: Tensor) -> float:
#         entropy_Ax = MatrixBasedRenyisEntropy.entropy(Kx)
#         entropy_Ay = MatrixBasedRenyisEntropy.entropy(Ky)
#         joint_entropy = MatrixBasedRenyisEntropy.jointEntropy(Kx, Ky)
#         return (entropy_Ax + entropy_Ay - joint_entropy)


#     # To remove!
#     @staticmethod
#     def matrix_estimator(x: Tensor, sigma: float) -> Tensor:
#         '''
#             Generates the 'A' matrix based on RBF kernel

#             @return 'A' matrix
#         '''
#         return TensorKernel.RBF(x, sigma) / len(x)

class TensorKernel:
    def RBF(x: Tensor, sigma: float) -> Tensor:
        '''
            Tensor Based Radial Basis Function (RBF) Kernel

            @param x: Tensor shape (n, features) or (batch, n, features)
            @param sigma
        '''
        assert x.ndim > 1 and x.ndim < 4, "The dimension of X must be 2 or 3"
        pairwise_difference = (torch.unsqueeze(x,x.ndim-1) - torch.unsqueeze(x,x.ndim-2))**2
        distance = torch.sum(pairwise_difference, dim=x.ndim)
        return torch.exp(-distance / (2*(sigma**2)) )

class KernelBasedEstimator():
    @staticmethod
    def entropy(A: Tensor) -> float:
        eigval, _ = torch.linalg.eigh(A)        
        epsilon = 1e-8
        eigval = eigval.abs() + epsilon 
        return -torch.sum(eigval*(torch.log2(eigval)), dim=eigval.ndim-1)

    @staticmethod
    def jointEntropy(Kx: Tensor, *args: Tensor) -> float:
        '''
            Parameters
            ----------
                Kx: Tensor

                args: More tensors!!!
        '''
        A = Kx.clone()
        for val in args:
            A = A * val
        
        A = A/A.trace() if A.ndim == 2 else A/(torch.sum(A.diagonal(offset=0, dim1=-1, dim2=-2), dim=1).reshape(-1,1,1))
        return KernelBasedEstimator.entropy(A)

    @staticmethod
    def mutualInformation(Ax: Tensor, Ay: Tensor) -> float:
        entropy_Ax = KernelBasedEstimator.entropy(Ax)
        entropy_Ay = KernelBasedEstimator.entropy(Ay)
        joint_entropy = KernelBasedEstimator.jointEntropy(Ax, Ay)
        return (entropy_Ax + entropy_Ay - joint_entropy)

    # To remove!
    @staticmethod
    def matrix_estimator(x: Tensor, sigma: float) -> Tensor:
        '''
            Generates the 'A' matrix based on RBF kernel

            @return 'A' matrix
        '''
        Kx = TensorKernel.RBF(x, sigma)
        return Kx / Kx.size(-1)
