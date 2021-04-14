from torch import nn
from abc import ABC, abstractmethod
from IPDL import MatrixEstimator

class MatrixOptimizer(ABC):
    '''
        Base class of matrix estimator optimizers which is used to 
        update the sigma values in MatrixEstimator class. This class
        is abstract and it is  necessary to implement _optimizer(*args)
        method in the inherited classes .
    '''
    def __init__(self, model: nn.Module):
        self.matrix_estimators = []
        for module in model.modules():
            if isinstance(module, (MatrixEstimator)):
                self.matrix_estimators.append(module)
    
    def step(self, *args) -> None:
        if args:
            self._optimize(*args)
        else:
            self._optimize(None)
        
        
    @abstractmethod
    def _optimize(self, *args) -> None:
        pass