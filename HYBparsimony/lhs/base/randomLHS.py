# -*- coding: utf-8 -*-

from HYBparsimony.lhs.util import findorder
from HYBparsimony import order
import numpy as np

   
def randomLHS(n, k, bPreserveDraw=False, seed=None):
    r"""Construct a random Latin hypercube design

    `randomLHS(4,3)` returns a 4x3 matrix with each column constructed as follows: A random per-mutation of (1,2,3,4) 
    is generated, say (3,1,2,4) for each of K columns. Then a uniform randomnumber is picked from each indicated 
    quartile. In this example a random number between `0.5` and `0.75` is chosen, then one between `0 `and `0.25`, then one 
    between `0.25` and `0.5`, finally one between `0.75` and `1`.

    Parameters
    ----------
    n : int
        The number of rows or samples. 
    k : int
        The number of columns or parameters/variables.
    bPreserveDraw : bool, optional
        Should the draw be constructed so that it is the same for variable numbers of columns?. Default `False`
    seed : int, optional
        Random seed. Default `None`.

    Returns
    -------
    numpy.array
        A `numpy.array` of `float` with shape `(n, k)`.
    """
    
    if n < 1 or k < 1:
        raise Exception("nsamples are less than 1 (n) or nparameters less than 1 (k)")

    if seed:
        np.random.seed(seed)

    result = np.zeros(n*k).reshape((n, k))
        
    randomunif1 = np.empty(n).astype(np.double)

    if bPreserveDraw:
        
        randomunif2 = np.empty(n).astype(np.double)
        for jcol in range(k):
            
            for irow in range(n):
                randomunif1[irow] = np.random.uniform(low=0, high=1)
            for irow in range(n):
                randomunif2[irow] = np.random.uniform(low=0, high=1)


            orderVector = order(randomunif1)
            for irow in range(n):
                result[irow,jcol] = orderVector[irow] + randomunif2[irow]
                result[irow,jcol] =  result[irow,jcol] / np.double(n)

    else:
        randomunif2 = np.empty(n*k).astype(np.double)
        for jcol in range(k):
            for irow in range(n):
                randomunif1[irow] = np.random.uniform(low=0, high=1)

            orderVector = order(randomunif1)
            for irow in range(n):
                result[irow,jcol] = orderVector[irow]

        for i in range(n*k):
            randomunif2[i] = np.random.uniform(low=0, high=1)

        randomunif2 = randomunif2.reshape((n, k))
        for jcol in range(k):

            for irow in range(n):
                result[irow,jcol] = result[irow,jcol] + randomunif2[irow, jcol]
                result[irow,jcol] = result[irow,jcol] / np.double(n)

    return result

def randomLHS_int(n, k, seed=None):
    r"""Construct a random Latin hypercube design

    `randomLHS(4,3)` returns a 4x3 matrix with each column constructed as follows: A random per-mutation of (1,2,3,4) 
    is generated, say (3,1,2,4) for each of K columns. Then a uniform randomnumber is picked from each indicated 
    quartile. In this example a random number between `0.5` and `0.75` is chosen, then one between `0 `and `0.25`, then one 
    between `0.25` and `0.5`, finally one between `0.75` and `1`.

    Parameters
    ----------
    n : int
        The number of rows or samples. 
    k : int
        The number of columns or parameters/variables.
    seed : int, optional
        Random seed. Default `None`.

    Returns
    -------
    numpy.array
        A `numpy.array` of `int` with shape `(n, k)`.
    """

    if seed:
        np.random.seed(seed)

    result = np.empty((n, k)).astype(np.int32)
    randomunif1 = np.empty(n).astype(np.double)
    for jcol in range(k):
    
        for irow in range(n):
            randomunif1[irow] = np.random.uniform(low=0, high=1)
        
        orderVector = findorder(randomunif1)
        for irow in range(n):
            result[irow,jcol] = orderVector[irow]

    return result
