# -*- coding: utf-8 -*-

import numpy as np
from HYBparsimony.lhs import randomLHS_int
from HYBparsimony.lhs.util import calculateSOptimal, calculateDistance, runifint, runif_std, findorder_zero

def geneticLHS(n, k, pop=100, gen=4, pMut=0.1, criterium="S", seed=None):
    r"""Latin Hypercube Sampling with a Genetic Algorithm

    Draws a Latin Hypercube Sample from a set of uniform distributions for use in creating 
    a LatinHypercube Design.  This function attempts to optimize the sample with respect 
    to the S optimalitycriterion through a genetic type algorithm.

    Parameters
    ----------
    n : int
        The number of rows or samples. 
    k : int
        The number of columns or parameters/variables.
    pop : int, optional
        The number of designs in the initial population. Default `100`.
    gen : int, optional
        The number of generations over which the algorithm is applied. Default `4`.
    pMut : float, optional
        The probability with which a mutation occurs in a column of the progeny. Default `0.1`.
    criterium : str, {'S', 'Maximin'}, optional
        The optimality criterium of the algorithm. Default is `'S'`. Maximin is also supported.
    seed : int, optional
        Random seed. Default `None`.

    Returns
    -------
    numpy.array
        A `numpy.array` of `float` with shape `(n, k)`.
    """

    if n < 1 or k < 1:
        raise Exception("nsamples are less than 1 (n) or nparameters less than 1 (k)")
    
    m_n = n
    m_k = k

    result = np.empty((m_n, m_k)).astype(np.double)
    if gen < 1 or pop < 1:
        raise Exception("pop, and gen should be integers greater than 0")
    
    m_pop = pop
    m_gen = gen
    
    if pMut <= 0 or pMut >= 1:
        raise Exception("pMut should be between 0 and 1")
    if m_pop % 2 != 0:
        raise Exception("pop should be an even number")

    if seed:
        np.random.seed(seed)

    A = np.empty(m_pop).astype(object)
    
    for i in range(m_pop):
        # // fill A with random hypercubes
        A[i] = randomLHS_int(m_n, m_k)
    
    for _ in range(m_gen):
        B = np.empty(m_pop).astype(np.double)
        for i in range(m_pop):
            if criterium == "S":
                B[i] = int(calculateSOptimal(A[i]))
            elif criterium == "Maximin":
                dist = calculateDistance(A[i]).astype(np.int32)[0]
                # // we want to find the minimum distance element, but there are zeros in the dist matrix
                distnonzero = []
                for mit in dist:
                    if mit > 0.0:
                        distnonzero.append(mit)

                distnonzero = np.array(distnonzero) 
                it = np.min(distnonzero) if distnonzero.shape[0]>0 else 0.0
                B[i] = it
            
            else: 
                raise Exception(f"Criterium not recognized: S and Maximin are available: {criterium} was provided.")

        # // H is used as an index on vector of matrices, A, so it should be using zero based order
        H = findorder_zero(B)
        posit = int(np.max(B) - B[0])
        J = np.empty(m_pop).astype(object)
        # // the first half of the next population gets the best hypercube from the first population
        for i in range(int(m_pop / 2)):
            J[i] = A[posit]
        
        if m_pop / 2 == 1:
            break
        
        # // the second half of the next population gets the decreasingly best hypercubes from the first population
        for i in range(int(m_pop / 2)):
            J[int(i + m_pop / 2)] = A[int(H[i])]
        
        # // skip the first best hypercube in the next generation
        # // in the others in the first half of the population, randomly permute a column from the second half into the first half
        for i in range(1, int(m_pop / 2)):
            temp1 = runifint(0, m_k-1).astype(np.int32)
            temp2 = runifint(0, m_k-1).astype(np.int32)
            for irow in range(m_n):
                J[i][irow, temp1] = J[int(i + m_pop / 2)][irow, temp2]
            
        # // for the second half of the population, randomly permute a column from the best hypercube
        for i in range(int(m_pop / 2), m_pop):
            temp1 = runifint(0, m_k-1).astype(np.int32)
            temp2 = runifint(0, m_k-1).astype(np.int32)
            for irow in range(m_n):
                J[i][irow, temp1] = A[posit][irow, temp2]
            
        # // randomly exchange two numbers in pMut percent of columns
        for i in range(1, m_pop):
            y = runif_std(m_k)
            for j in range(m_k):
                if y[j] <= pMut:
                    z = runifint(0, m_n-1, 2).astype(np.int32)
                    a = J[i][z[0], j]
                    b = J[i][z[1], j]
                    J[i][z[0], j] = b
                    J[i][z[1], j] = a
                
        # // put all of J back into A to start the next round
        A = J
    
    eps = runif_std(m_n * m_k)
    count = 0
    for j in range(m_k):
        for i in range(m_n):
            result[i,j] = (np.double(J[0][i,j]) - 1.0 + eps[count]) / np.double(m_n)
            count += 1
        
    return result


