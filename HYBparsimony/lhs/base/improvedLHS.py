# -*- coding: utf-8 -*-

import numpy as np
from HYBparsimony.lhs.util import initializeAvailableMatrix, convertIntegerToNumericLhs


def _improvedLHS( n,  k, dup=1, seed=None):

    if seed:
        np.random.seed(seed)
    nsamples = n
    nparameters = k
    duplication = dup
    result = np.empty((k, n)).astype(int)
    #  ********** matrix_unsafe<int> m_result = matrix_unsafe<int>(nparameters, nsamples, result);
    #  the length of the point1 columns and the list1 vector 
    size = duplication * (nsamples - 1)
    
    #  create memory space for computations 
    point1 = np.empty((nparameters, size))
    list1 = np.empty(size)
    vec = np.empty(nparameters)
    #  optimum spacing between points 
    opt = np.double(nsamples) / ( np.power(np.double(nsamples), (1.0 / np.double(nparameters))))
    #  the square of the optimum spacing between points 
    opt2 = opt * opt
    #  initialize the avail matrix 
    avail = initializeAvailableMatrix(nparameters, nsamples)

    # * come up with an array of K integers from 1 to N randomly
    # * and put them in the last column of result
    for irow in range(nparameters):
        result[irow, nsamples-1] = int(np.floor(np.random.uniform(low=0, high=1) * np.double(nsamples) + 1.0))
    
    # * use the random integers from the last column of result to place an N value
    # * randomly through the avail matrix
    for irow in range(nparameters):
        avail[irow, int(result[irow, nsamples-1] - 1)] = nsamples
    
    #  move backwards through the result matrix columns.
    for ucount in range(nsamples - 1, 0, -1):
        # unsigned int ucount = static_cast<unsigned int>(count);
        for irow in range(nparameters):
            for jcol in range(duplication):
                #  create the list1 vector 
                for j in range(ucount):
                    list1[j + ucount*jcol] = avail[irow, j]
                
            #  create a set of points to choose from. Note, need to use int
            #  Note: can't do col = count*duplication - 1; col >= 0 because it throws a warning at W4 
            for ujcol in range(ucount * duplication, 0, -1):
                # unsigned int ujcol = static_cast<unsigned int>(jcol);
                point_index = int(np.floor(np.random.uniform(low=0, high=1) * np.double(ujcol)))
                point1[irow, ujcol-1] = list1[point_index]
                list1[point_index] = list1[ujcol-1]
        
        min_all = np.finfo(np.double).tiny
        best = 0
        for jcol in range(duplication * ucount - 1):
            min_candidate = 4294967295
            for j in range(nsamples):
                distSquared = 0
        
                # * find the distance between candidate points and the points already
                # * in the sample
                for kindex in range(nparameters):
                    vec[kindex] = point1[kindex, jcol] - result[kindex, j]
                    distSquared += (vec[kindex] + 10e-10)**2 # vec[kindex] * vec[kindex]   # Da un warning
                
                # * if the distSquard value is the smallest so far place it in
                # * min candidate
                if min_candidate > distSquared:
                
                    min_candidate = distSquared
                
            # * if the difference between min candidate and opt2 is the smallest so
            # * far, then keep that point as the best.
            if np.abs(np.double(min_candidate) - opt2) < min_all:
                min_all = np.abs(np.double(min_candidate) - opt2)
                best = jcol
        
        #  take the best point out of point1 and place it in the result 
        for irow in range(nparameters):
            result[irow, ucount - 1] = point1[irow, best]
        
        #  update the numbers that are available for the future points 
        for irow in range(nparameters):
            for jcol in range(nsamples):
                if avail[irow, jcol] == result[irow, ucount - 1]:
                    avail[irow, jcol] = avail[irow, ucount-1]
                
    # * once all but the last points of result are filled in, there is only
    # * one choice left
    for jrow in range(nparameters):
        result[jrow, 0] = avail[jrow, 0]
    
    
    return result.transpose()

def improvedLHS(n, k, dup=1, seed=None):
    r"""Improved Latin Hypercube Sample

    Draws a Latin Hypercube Sample from a set of uniform distributions for use in creating a Latin
    Hypercube Design. This function attempts to optimize the sample with respect to 
    an optimum euclidean distance between design points.

    Parameters
    ----------
    n : int
        The number of rows or samples. 
    k : int
        The number of columns or parameters/variables.
    dup : int, optional
        A factor that determines the number of candidate points used in the search. 
        A multiple of the number of remaining points than can be added. Default `1`.
    seed : int, optional
        Random seed. Default `None`.

    Returns
    -------
    numpy.array
        A `numpy.array` of `float` with shape `(n, k)`.
    """

    return np.random.rand(k)[np.newaxis, :] if n == 1 else convertIntegerToNumericLhs(_improvedLHS(n, k, dup, seed))