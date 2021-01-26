from torch import Tensor, nn
from .MatrixEstimator import MatrixEstimator

class MatrixOptimizer():
    def __init__(self, model: nn.Module, beta=0.5, n_sigmas=300):
        if not(0 <= beta <= 1):
            raise Exception('beta must be in the range [0, 1]')

        self.matrix_estimators = []
        # First element corresponds to input A matrix and last element
        # is the output A matrix
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrix_estimators.append(module)

        self.beta = beta
        self.n_sigmas = n_sigmas

        #just for debugging
        self.sigma_prev = [-1] * len(self.matrix_estimators)
        for idx, matrix_estimator in enumerate(self.matrix_estimators):
            self.sigma_prev[idx] = [matrix_estimator.get_sigma()]

    def step(self, Ky: Tensor) -> None:
        self.optimize(Ky)
    
    def optimize(self, Ky: Tensor) -> None:
        '''
            This function is used in orter to obtain the optimal kernel width for
            an T DNN layer

            @param layer_output
            @param n_sigmas: number of possible sigma values

            [Descripción del procedimiento]
        '''
        device = Ky.device
        for idx, matrix_estimator in enumerate(self.matrix_estimators):
            activation = nn.Softmax() if idx == len(self.matrix_estimators)-1 else nn.Identity()

            x = activation(matrix_estimator.x).to(device)
            sigma_values = self.getPossibleSigmaValues(x, self.n_sigmas)

            Kt = list( map(lambda sigma: TensorKernel.RBF(x, sigma), sigma_values) )    
            
            loss = np.array( list( map(lambda k: self.kernelAligmentLoss(k, Ky), Kt) ) )
            best_sigma = sigma_values[ np.argwhere(loss == loss.max()).item(0) ]
            
            best_sigma = ( (self.beta*best_sigma) + ((1-self.beta)*matrix_estimator.get_sigma()) )
            
            #Just for debugging
            self.sigma_prev[idx].append(best_sigma)

            matrix_estimator.set_sigma(best_sigma)

    
    def kernelAligmentLoss(self, Kx: Tensor, Ky: Tensor) -> float:
        '''
            Kernel Aligment Loss Function.

            This function is used in order to obtain the optimal sigma parameter from
            RBF kernel.  
        '''
        return (torch.sum(Kx*Ky)/(torch.norm(Kx) * torch.norm(Ky))).item()

    
    def getPossibleSigmaValues(self, x: Tensor, n=100) -> Tensor:
        '''
            Obtener los sigma values para el optimizador
        '''
        distance = torch.cdist(x,x)
        distance = distance[torch.triu(torch.ones(distance.shape, dtype=torch.bool), diagonal=1)] 
        return torch.linspace(0.1, 10*distance.mean(), n).tolist()