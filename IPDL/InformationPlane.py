import numpy as np
from torch import Tensor, nn
from abc import ABC, abstractmethod
from pandas import DataFrame, MultiIndex
from .MatrixEstimator import MatrixEstimator
from .InformationTheory import MatrixBasedRenyisEntropy as renyis

class InformationPlane(ABC):
    def __init__(self, model: nn.Module):
        self.matrix_estimators = []
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrix_estimators.append(module)

        self.Ixt = [] # Mutual Information I(X,T)
        self.Ity = [] # Mutual Information I(T,Y)

    def getMutualInformation(self, moving_average_n = 0):
        from .utils import moving_average as mva

        if moving_average_n == 0:
            return self.Ixt, self.Ity
        else:
            filter_Ixt = list(map(lambda Ixt: mva(Ixt, moving_average_n), self.Ixt))
            filter_Ity = list(map(lambda Ity: mva(Ity, moving_average_n), self.Ity))
            return filter_Ixt, filter_Ity

    def to_df(self):
        Ixt = np.array(self.Ixt)
        Ity = np.array(self.Ity)
        index_names = [
                list(map(lambda x: 'Layer {}'.format(x), np.repeat(np.arange(len(Ixt)), 2)+1 )),
                ['Ixt', 'Ity']*len(Ixt)
            ]
        
        tuples = list(zip(*index_names))
        index = MultiIndex.from_tuples(tuples)
        MI = np.zeros((Ixt.shape[1], Ixt.shape[0]*2), dtype=np.float)
        MI[:, 0::2] = Ixt.T
        MI[:, 1::2] = Ity.T

        return DataFrame(MI, columns=index)


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

        return list(map(lambda x: x[-1], self.Ixt)), list(map(lambda x: x[-1], self.Ity))


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
        '''
            Compute the Mutual Information. 

            return two list which represents the Ixt and Ity
            in the different layers.
        '''
        Ay = self.matrix_estimators[-1].get_matrix()

        for idx, matrix_estimator in enumerate(self.matrix_estimators[0:-1]):
            self.Ixt[idx].append(renyis.mutualInformation(Ax, matrix_estimator.get_matrix()).cpu())
            self.Ity[idx].append(renyis.mutualInformation(matrix_estimator.get_matrix(), Ay).cpu())

        return list(map(lambda x: x[-1], self.Ixt)), list(map(lambda x: x[-1], self.Ity))