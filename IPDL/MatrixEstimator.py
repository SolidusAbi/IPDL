import torch
from torch import Tensor, nn
from .InformationTheory import TensorKernel

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
            self.x = x.detach().flatten(1).cpu() # To CPU in order to save memory on GPU

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
            return (TensorKernel.RBF(activation(self.x).to(device), self.sigma) / n)
        else:
            return (TensorKernel.RBF(self.x.to(device), self.sigma) / n)

    def __repr__(self) -> str:
        return "MatrixEstimator(sigma={:.2f})".format(self.sigma)



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