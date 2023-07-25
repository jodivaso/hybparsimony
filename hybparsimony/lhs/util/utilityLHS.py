# -*- coding: utf-8 -*-

import numpy as np
from .bclib import inner_product

def isValidLHS_int(matrix):
    for jcol in range(matrix.shape[1]):
        total = 0
        for irow in range(matrix.shape[0]):
            total = total + matrix[irow, jcol]
        if not total == int(matrix.shape[0] * (matrix.shape[0] + 1) / 2):
        
            return False
        
    return True

def isValidLHS(matrix):

    n = matrix.shape[0]
    k = matrix.shape[1]
    resultint = np.empty((n, k))
    # for (;it != result.end(); ++it, ++iti) # Itera sobre filas, fichero matrix.h
    for i in range(matrix.shape[0]):
        resultint[i] = 1 + (np.floor(np.double(n) * (matrix[i]))).astype(np.int32)

    for jcol in range(resultint.shape[1]):
        total = 0
        for irow in range(resultint.shape[0]):
            total = total + resultint[irow, jcol]
        if not total == int(resultint.shape[0] * (resultint.shape[0] + 1) / 2):
        
            return False
        
    return True     

def initializeAvailableMatrix(i, j):
    dev = np.empty((i, j)).astype(np.double)
    for irow in range(i):
    
        for jcol in range(j):
        
            dev[irow, jcol] = np.double(jcol + 1)
        
    return dev

def runif_std(n):
    dev = np.empty(n).astype(np.double)
    for i in range(n):
        dev[i] =np.random.uniform(low=0, high=1)
    return dev

def convertIntegerToNumericLhs(intMat):
  n = intMat.shape[0]
  k = intMat.shape[1]
  result = np.empty((n, k))
  eps = np.random.rand(n*k)
  counter = 0
  # // I think this is right (iterate over rows within columns
  for col in range(k):
      for row in range(n):
        result[row, col] = np.double(intMat[row, col] - 1) + eps[counter]
        result[row, col] = result[row, col] / np.double(n)
        counter+=1
  return result


def sumInvDistance(a): 
	return np.sum(calculateDistance(a)[::-1]) # equals to accumulate
    

def calculateDistanceSquared(a, b):

    if a.shape != b.shape:
        raise Exception("Inputs of a different size")
    
    return inner_product(a, b, np.double(0), lambda a, b: a+b, lambda x, y: (x-y) * (x-y))

def calculateDistance(mat):
    m_rows = mat.shape[0]
    result = np.empty((m_rows, m_rows)).astype(np.double)
    for i in range(m_rows - 1):
        for j in range(i+1, m_rows):
            result[i,j] = np.sqrt(np.double(calculateDistanceSquared(mat[i,:], mat[j,:])))
    return result
        
    
def calculateSOptimal(mat):
    return 1.0 / sumInvDistance(mat)


def runifint(a, b, n=None):
    if not n:
        return  a + (np.floor((np.random.uniform(low=0, high=1) * (np.double(b) + 1.0 - np.double(a)))))
    else:
        result = np.empty(n).astype(np.double)
        r = runif_std(n)
        for i in range(n):
            result[i] = a + (np.floor(r[i] * (b + 1.0 - b + 1.0 - a)))
        return result

    