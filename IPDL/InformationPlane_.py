# from IPDL import MatrixBasedRenyisEntropy as renyis
from torch import Tensor, nn
from .MatrixEstimator import MatrixEstimator
from . import MatrixBasedRenyisEntropy as renyis

from .utils import moving_average as mva

class InformationPlane():
    def __init__(self):
        self.Ixt = [] # Mutual Information I(X,T)
        self.Ity = [] # Mutual Information I(T,Y)

    def getMutualInformation(self, moving_average_n = 0):
        if moving_average_n == 0:
            return self.Ixt, self.Ity
        else:
            filter_Ixt = list(map(lambda Ixt: mva(Ixt, moving_average_n), self.Ixt))
            filter_Ity = list(map(lambda Ity: mva(Ity, moving_average_n), self.Ity))
            return filter_Ixt, filter_Ity

    def computeMutualInformation(self, Ax: Tensor, Ay: Tensor):
        raise NotImplementedError("Please Implement this method")


class ClassificationInformationPlane(InformationPlane):
    '''
        # Pass a list of tensor which contents the matrices in order to calculate the
        # MutualInformation 

        IP implementaiton that works for classification problems.
    '''

    def __init__(self, model: nn.Module, use_softmax=True):
        '''
            @param model: model where 
            @param softmax: include a softmax layer at the end of the model. It is usefull 
                if your model does not contain this layer. 
        '''
        super(ClassificationInformationPlane, self).__init__()
        self.matrices_per_layers = []
        self.use_softmax = use_softmax
      
        # First element corresponds to input A matrix and last element
        # is the output A matrix
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrices_per_layers.append(module)

        for i in range(len(self.matrices_per_layers)):
            self.Ixt.append([])
            self.Ity.append([])
    
    def computeMutualInformation(self, Ax: Tensor, Ay: Tensor):
        for idx, matrix_estimator in enumerate(self.matrices_per_layers):
            activation = nn.Softmax() if self.use_softmax and idx == len(self.matrices_per_layers)-1 else None

            self.Ixt[idx].append(renyis.mutualInformation(Ax, matrix_estimator.get_matrix(activation)).cpu())
            self.Ity[idx].append(renyis.mutualInformation(matrix_estimator.get_matrix(activation), Ay).cpu())

'''
    No tested!! (see IPAE)
 '''
class AutoEncoderInformationPlane(InformationPlane):
    def __init__(self, model: nn.Module):
        super(AutoEncoderInformationPlane, self).__init__()
        self.matrices_per_layers = []

        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrices_per_layers.append(module)
        
        for i in range(len(self.matrices_per_layers)-2):
            self.Ixt.append([])
            self.Ity.append([])

    def computeMutualInformation(self):
        Ax = self.matrices_per_layers[0].A
        Ay = self.matrices_per_layers[-1].A

        for idx, matrix_estimator in enumerate(self.matrices_per_layers[1:-1]):
            self.Ixt[idx].append(renyis.mutualInformation(Ax, matrix_estimator.get_matrix()).cpu())
            self.Ity[idx].append(renyis.mutualInformation(matrix_estimator.get_matrix(), Ay).cpu())