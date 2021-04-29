import torch
from torch import Tensor, nn
from torch.nn.functional import batch_norm
from .InformationTheory import TensorKernel

class MatrixEstimator(nn.Module):
    def __init__(self, sigma = 0.1, requires_optim = True):
        super(MatrixEstimator, self).__init__()
        
        self.sigma = nn.Parameter(torch.tensor(sigma), requires_grad=False)
        self.x = torch.rand((10, 1)) # Dummy!
        self.requires_optim = requires_optim

    def set_sigma(self, sigma: float) -> None:
        if self.requires_optim:
            self.sigma.data = torch.tensor(sigma, device=self.sigma.device)

    def get_sigma(self) -> float:
        return self.sigma.data.item()

    def get_device(self) -> str:
        return self.sigma.device

    def forward(self, x: Tensor) -> Tensor:
        if not self.training:
            # Move to CPU just for saving memory on GPU
            # (dar una vuelta a ver si realmente vale la pena)
            self.x = x.detach().clone().cpu()

        return x

    def get_matrix(self, activation = None) -> Tensor:
        '''
            Return matrix A

            @param activation: you can chose any nn.Module if is necesary to apply an activation. 
                It is usefull if you need include an activation at the end of your model to compute 
                the Information Plane.
        '''
        device = self.sigma.device # To the device where parameters are located
        n = self.x.size(0)       
        x = self.x.flatten(1).to(device)
        if not(activation is None):
            x = activation(x)

        return (TensorKernel.RBF(x, self.sigma) / n)

    def __repr__(self) -> str:
        return "MatrixEstimator(sigma={:.2f}, requires_optim={})".format(self.sigma, self.requires_optim)



### TEST (Esto debería ponerlo en otro fichero) ###
def test():
    from torch import nn

    model = nn.Sequential(
        nn.Identity(),
        MatrixEstimator(8),
        nn.Identity(),
        MatrixEstimator(0.5),
        nn.Identity(),
        MatrixEstimator(0.100001)
    )
    model.eval()
    
    x = torch.rand(10, 10)
    y = model(x)

    Ax_0 = TensorKernel.RBF(x, sigma=8) / x.size(0)
    Ax_1 = TensorKernel.RBF(x, sigma=0.5) / x.size(0)
    Ax_2 = TensorKernel.RBF(x, sigma=0.100001) / x.size(0)

    case_0 = all((model[1].get_matrix() == Ax_0).flatten())
    print("Test 0: {}".format(case_0))

    case_1 = all((model[3].get_matrix() == Ax_1).flatten())
    print("Test 1: {}".format(case_1))

    case_2 = all((model[5].get_matrix() == Ax_2).flatten())
    print("Test 2: {}".format(case_2))

if __name__ == "__main__":
        test()