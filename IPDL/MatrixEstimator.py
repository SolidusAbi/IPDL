from torch import Tensor, nn
from . import TensorKernel

class MatrixEstimator(nn.Module):
    def __init__(self, sigma = 0.1):
        super(MatrixEstimator, self).__init__()
        
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.x = torch.rand((10, 1))

    def set_sigma(self, sigma: float) -> None:
        self.sigma.data = torch.tensor(sigma)

    def get_sigma(self) -> float:
        return self.sigma.data.item()

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            self.x = x.detach().cpu() # To CPU in order to save memory on GPU

        return x

    def get_matrix(self, activation= None) -> Tensor:
        '''
            Return matrix A

            @param activation: you can chose any nn.Module if is necesary to apply an activation. 
                It is usefull if you need include an activation at the end of your model to compute 
                the Information Plane.
        '''
        device = self.sigma.device # To the device where parameters are located
        n = self.x.size(0)
        
        if not(activation is None):
            return (TensorKernel.RBF(activation(self.x).flatten(1).to(device), self.sigma) / n)
        else:
            return (TensorKernel.RBF(self.x.flatten(1).to(device), self.sigma) / n)

    def __repr__(self) -> str:
        return "MatrixEstimator(sigma={:.2f})".format(self.sigma)