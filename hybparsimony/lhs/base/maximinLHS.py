# -*- coding: utf-8 -*-

import numpy as np
from hybparsimony.lhs.util import initializeAvailableMatrix, convertIntegerToNumericLhs

# result es lo que devolvemos

def _maximinLHS(n, k, dup=1, seed=None):

    if n < 1 or k < 1 or dup < 1:
        raise Exception("nsamples are less than 1 (n) or nparameters less than 1 (k) or duplication is less than 1")

    if seed:
        np.random.seed(seed)

    nsamples = n
    nparameters = k
    duplication = dup

    result = np.empty((k, n)).astype(np.double)

    # the length of the point1 columns and the list1 vector 
    leng = duplication * (nsamples - 1)

    # create memory space for computations 
    point1 = np.empty((nparameters, leng))
    list1 = np.empty(leng)
    vec = np.empty(nparameters)

    #  squared distance between corner (1,1,1,..) and (N,N,N,...) 
    squaredDistanceBtwnCorners = np.double(nparameters * (nsamples - 1) * (nsamples - 1))
    avail = initializeAvailableMatrix(nparameters, nsamples)

    # * come up with an array of K integers from 1 to N randomly
    # * and put them in the last column of result
    for irow in range(nparameters):
        result[irow, nsamples-1] = np.double(np.floor(np.random.uniform(low=0, high=1) * np.double(nsamples) + 1.0))
    
    # * use the random integers from the last column of result to place an N value
    # * randomly through the avail matrix
    # 
    for irow in range(nparameters):
        avail[irow,int(result[irow, nsamples - 1] - 1)] = int(nsamples)
    
    #  move backwards through the result matrix columns 
    for ucount in range(nsamples - 1, 0, -1):
        for irow in range(nparameters):
            for jcol in range(duplication):
                #  create the list1 vector 
                for j in range(ucount):              
                    list1[j + ucount*jcol] = avail[irow, j]
                
            #  create a set of points to choose from 
            for jcol in range(ucount * duplication, 0, -1):
                point_index = int(np.floor(np.random.uniform(low=0, high=1) * np.double(jcol)))
                point1[irow, jcol-1] = list1[point_index]
                list1[point_index] = list1[jcol - 1]
            
        minSquaredDistBtwnPts = np.finfo(np.double).tiny #DBL_MIN el menor float representable
        best = 0
        for jcol in range(duplication * ucount - 1):
            #  set min candidate equal to the maximum distance to start 
            minCandidateSquaredDistBtwnPts = int(np.ceil(squaredDistanceBtwnCorners))
            for  j in range(ucount, nsamples):
                distSquared = 0

                # * find the distance between candidate points and the points already
                # * in the sample
                for kindex in range(nparameters):
                    vec[kindex] = point1[kindex, jcol] - result[kindex, j]
                    distSquared = distSquared + (vec[kindex] * vec[kindex])
                
                # * if the distance squared value is the smallest so far, place it in the
                # * min candidate
                if minCandidateSquaredDistBtwnPts > distSquared:
                    minCandidateSquaredDistBtwnPts = distSquared
                
            # * if the candidate point is the largest minimum distance between points so
            # * far, then keep that point as the best.
            if np.double(minCandidateSquaredDistBtwnPts) > minSquaredDistBtwnPts:
                minSquaredDistBtwnPts = np.double(minCandidateSquaredDistBtwnPts)
                best = int(jcol)
            
        #  take the best point out of point1 and place it in the result 
        for irow in range(nparameters):
            result[irow, ucount-1] = point1[irow, best]

        #  update the numbers that are available for the future points 
        for irow in range(nparameters):
            for jcol in range(nsamples):
                if avail[irow, jcol] == result[irow, ucount-1]:
                    avail[irow, jcol] = avail[irow, ucount-1]
                
    # * once all but the last points of result are filled in, there is only
    # * one choice left
    for irow in range(nparameters):
        result[irow, 0] = avail[irow, 0]

    return result.transpose()


def maximinLHS(n, k, dup=1, seed=None):
    r"""Maximin Latin Hypercube Sample

    Draws a Latin Hypercube Sample from a set of uniform distributions for use in creating 
    a Latin Hypercube Design. This function attempts to optimize the sample by 
    maximizing the minium distance between design points (maximin criteria).

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
        An `n` by `n` Latin Hypercube Sample matrix with values uniformly distributed on `[0,1]`.
    """
    
    return np.random.rand(k)[np.newaxis, :] if n==0 else convertIntegerToNumericLhs(_maximinLHS(n, k, dup, seed))
