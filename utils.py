import sys
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
    return np.ones_like(array1.shape) - almost_equal(array1, array2, tol=tol)  #


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


def read_vertex_range_from_file(filename, start_at_incl=0, stop_at_excl=np.infty, update_freq=1e6, batch_size=10000, dtype='int16'):
    with open(filename, 'r') as f:
        # skip empty lines and commented lines at start of file
        line = f.readline()
        while line and (line[0] == '#' or not line.strip()):
            line = f.readline()

        # find out dimension of vertices
        d = len(line.split())
        Q = np.empty((0, d), dtype)
        batch = []
        i = 0
        len_at_last_update = 0
        while line and i < stop_at_excl:
            if line.strip():
                if i >= start_at_incl:
                    batch.append(list(map(int, line.split())))
                if len(batch) >= batch_size:
                    Q = np.r_[Q, batch].astype(dtype)
                    batch = []
                    if len(Q) > len_at_last_update + update_freq:
                        print("Loading vertices from file: %d elements till now" % len(Q), end='\r')
                        sys.stdout.flush()
                        len_at_last_update = len(Q)
                i += 1
            line = f.readline()
        Q = np.r_[Q, batch]
        print()
    return Q
