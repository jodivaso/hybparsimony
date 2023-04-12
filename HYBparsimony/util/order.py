# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd

def order(obj, kind='heapsort', decreasing = False, na_last = True):
    r"""Function to order vectors 

    This function is an overload of `numpy.argsort` sorting method allowing increasing 
    and decreasing ordering and allowing nan values to be placed at the end and at the beginning.

    Parameters
    ----------
    obj : numpy.array
        Array to order.
    kind : {'quicksort', 'mergesort', 'heapsort', 'stable'}, optional
        Sorting algorithm. The default is `heapsort`. Note that both 'stable' and 'mergesort' use timsort 
        under the covers and, in general, the actual implementation will vary with data type.
    decreasing : bool, optional
        If we want decreasing order.
    na_last : bool, optional
        For controlling the treatment of `NA's`. If `True`, missing values in the data are put last, if `False`, they are put first.
    """

    if not decreasing:
        if na_last:
            return obj.argsort(kind=kind)
        else:
            na = np.count_nonzero(np.isnan(obj))
            aux = obj.argsort(kind=kind)
            return np.concatenate([aux[-na:], aux[:-na]])
    else:
        if not na_last:
            return obj.argsort(kind=kind)[::-1]
        else:
            na = np.count_nonzero(pd.isnull(obj))
            aux = obj.argsort(kind=kind)[::-1]
            return np.concatenate([aux[na:], aux[:na]])