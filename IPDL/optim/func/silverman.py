import math
from functools import reduce

def silverman_optimize(x, gamma, normalize):
    '''
        Silverman's rule of thung for Gaussian kernels.

        Eq:
        \sigma = \gamma N^{ ({-1}/(4+d)) }
        
        Parameters
        ----------
            x: torch.Tensor
                Input tensor
            gamma: float
                Gamma parameter
            normalize: bool
                Apply dimensional normalization proposed on "On the Information 
                Plane of Autoencoders".
                    
                    \sigma' = \sigma * \sqrt{d}
    '''

    n = x.size(0)
    d = x.size(1) if len(x.shape) == 2 else reduce(lambda x, y: x*y, x.shape[1:])
    
    sigma = gamma * (n ** (-1 / (4 + d)))
    if normalize:
        sigma = sigma * math.sqrt(d)

    return sigma