# from IPDL import MatrixBasedRenyisEntropy as renyis
from torch import Tensor, nn
from abc import ABC, abstractmethod
from .MatrixEstimator import MatrixEstimator
from .InformationTheory import MatrixBasedRenyisEntropy as renyis

from .utils import moving_average as mva

class InformationPlane(ABC):
    def __init__(self, model: nn.Module):
        self.matrix_estimators = []
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrix_estimators.append(module)

        self.Ixt = [] # Mutual Information I(X,T)
        self.Ity = [] # Mutual Information I(T,Y)

    def getMutualInformation(self, moving_average_n = 0):
        if moving_average_n == 0:
            return self.Ixt, self.Ity
        else:
            filter_Ixt = list(map(lambda Ixt: mva(Ixt, moving_average_n), self.Ixt))
            filter_Ity = list(map(lambda Ity: mva(Ity, moving_average_n), self.Ity))
            return filter_Ixt, filter_Ity

    @abstractmethod
    def computeMutualInformation(self, *args):
        pass


class ClassificationInformationPlane(InformationPlane):
    '''
        # Pass a list of tensor which contents the matrices in order to calculate the
        # MutualInformation 

        IP implementaiton that works for classification problems.
    '''

    def __init__(self, model: nn.Module, use_softmax=True):
        '''
            @param model: model which contains matrix estimators
            @param use_softmax: include a softmax layer at the end of the model. It is usefull 
                if your model does not contain this layer.
        '''
        super(ClassificationInformationPlane, self).__init__(model)

        self.use_softmax = use_softmax

        for i in range(len(self.matrix_estimators)):
            self.Ixt.append([])
            self.Ity.append([])
    
    def computeMutualInformation(self, Ax: Tensor, Ay: Tensor):
        for idx, matrix_estimator in enumerate(self.matrix_estimators):
            activation = nn.Softmax(dim=1) if self.use_softmax and idx == len(self.matrix_estimators)-1 else None

            self.Ixt[idx].append(renyis.mutualInformation(Ax, matrix_estimator.get_matrix(activation)).cpu())
            self.Ity[idx].append(renyis.mutualInformation(matrix_estimator.get_matrix(activation), Ay).cpu())


class AutoEncoderInformationPlane(InformationPlane):
    '''
       Computes Mutual Information to generate a Information Plane for AutoEncoders architectures.

       The matrix Ay is directly the model's output.
    '''
    def __init__(self, model: nn.Module):
        super(AutoEncoderInformationPlane, self).__init__(model)
        
        for i in range(len(self.matrix_estimators)-1):
            self.Ixt.append([])
            self.Ity.append([])

    def computeMutualInformation(self, Ax: Tensor):
        Ay = self.matrix_estimators[-1].get_matrix()

        for idx, matrix_estimator in enumerate(self.matrix_estimators[0:-1]):
            self.Ixt[idx].append(renyis.mutualInformation(Ax, matrix_estimator.get_matrix()))
            self.Ity[idx].append(renyis.mutualInformation(matrix_estimator.get_matrix(), Ay))