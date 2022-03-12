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
    return np.ones_like(array1.shape) - almost_equal(array1, array2, tol=tol)#


def eliminate_duplicate_rows(array):
    """ Returns array where duplicate rows have been eliminated. Contrary to np.unique(array, axis=0), this does NOT also sort the array.
     (However, I can't guarantee that, IF there are duplicates, the first occurrence of each duplicate row is taken.) """
    unique_and_sorted, indices = np.unique(array, axis=0, return_index=True)
    indices_sorted = np.sort(indices)
    print(indices)
    return np.array([array[i] for i in indices_sorted])

def assert_soft(statement):
    try:
        assert statement
    except AssertionError as e:
        print(e)


def approximate(vector, allowed_values):
    assert len(vector.shape) == 1  # Could easily generalise to not require this if necessary
    max_dev, argmax_dev = -1, -1
    approx_vector = np.zeros_like(vector)
    for i in range(len(vector)):
        closest_matching_index = np.argmin([abs(vector[i] - value) for value in allowed_values])
        approx_vector[i] = allowed_values[closest_matching_index]
        dev = vector[i] - allowed_values[closest_matching_index]
        if abs(dev) > max_dev:
            max_dev = abs(dev)
            argmax_dev = i
    print('Largest error in utils.approximate(): %s at index %i' % (str(max_dev), argmax_dev))
    return approx_vector