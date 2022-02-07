import time

import numpy as np


def timer(fun):
    start = time.time()
    fun()
    end = time.time()
    print(end - start)


def almost_equal(array1, array2, tol=1e-16):
    assert array1.shape == array2.shape
    return np.equal(array1, array2, out=np.ones_like(array1), where=np.abs(array1 - array2) >= tol)


def not_almost_equal(array1, array2, tol=1e-16):
    return np.ones_like(array1.shape) - almost_equal(array1, array2, tol=tol)