from torch import Tensor
from .InformationTheory import TensorKernel

def matrix_estimator(x: Tensor, sigma: float = .1):
    Kx = TensorKernel.RBF(x, sigma)
    Ax = Kx/x.size(0)
    return Kx, Ax