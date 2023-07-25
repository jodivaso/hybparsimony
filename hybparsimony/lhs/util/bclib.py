# -*- coding: utf-8 -*-

import numpy as np

# Ya estaría implementadas

def findorder_zero(v):
    
    # create a vector of pairs to hold the value and the integer rank
    p = np.empty(v.shape, dtype=object)
    for i, vi in enumerate(v):
        p[i] = (vi, i)
    p = np.array(list(sorted(p, key=lambda a: a[0])))
    
    return np.array(list(map(lambda x: x[1], p)))


def findorder(v):
    order = findorder_zero(v)
    for i in range(len(order)):
        order[i] += 1
    return order
 

def inner_product(a, b, init=0, op1=lambda a, b: a + b, op2=lambda a, b: a * b):
    for x, y in zip(a, b):
        init = op1(init, op2(x, y))
    return init
