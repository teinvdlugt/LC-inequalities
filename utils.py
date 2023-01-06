import itertools
import sys
import time

import numpy as np
import vector_space_utils as vs


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
        while line and (line.strip()[0] == '#' or not line.strip()):
            line = f.readline()

        # find out dimension of vertices
        d = len(line.split())
        Q = np.empty((0, d), dtype)
        batch = []
        i = 0
        len_at_last_update = 0
        while line and i < stop_at_excl:
            if line.strip() and not line.strip()[0] == '#':
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
        if len(Q) >= batch_size:
            print()
    return Q



def generate_dictionary_of_full_probs_in_NSS_coords(filename=None):
    with open(filename, 'w') as f:
        NtoF = vs.construct_NSS_to_full_h(8, 2, 4, 2).astype('int8')
        for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
            i = vs.concatenate_bits(a1, a2, c, b, x1, x2, y)
            f.write('p(%d%d%d%d|%d%d%d): ' % (a1, a2, c, b, x1, x2, y) + ' '.join(map(str, -NtoF[i])) + '\n')


def write_cor_to_file(cor, filename):
    with open(filename, 'w') as f:
        f.write('p(a1 a2 c b | x1 x2 y)\n')
        for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
            i = vs.concatenate_bits(a1, a2, c, b, x1, x2, y)
            f.write('p(%d%d%d%d|%d%d%d): %s' % (a1, a2, c, b, x1, x2, y, str(cor[i])) + '\n')


def filter_row_file(input_filename, output_filename, constraint):
    """ Creates a new file at output_filename that contains all integer rows of input_filename that satisfy constraint. It leaves
    blank lines where input_filename has blank lines, and ignores badly formatted lines in input_filename.
     :param constraint: a function taking integer vectors to booleans."""
    bad_line_count = 0
    constraint_satisfied_count = 0
    total_rows_processed_count = 0
    with open(input_filename, 'r') as input_file:
        with open(output_filename, 'w') as output_file:
            line = input_file.readline()
            while line:
                if not line.strip():
                    # blank line; let's also put one in output_file
                    output_file.write('\n')
                else:
                    try:
                        row = list(map(int, line.split()))
                        if constraint(row):
                            output_file.write(line)
                            constraint_satisfied_count += 1
                        total_rows_processed_count += 1
                    except:
                        bad_line_count += 1

                print("%d / %d good rows; %d bad lines" % (constraint_satisfied_count, total_rows_processed_count, bad_line_count), end='\r')
                sys.stdout.flush()
                line = input_file.readline()
    print("Finished with %d / %d good rows; %d bad lines" % (constraint_satisfied_count, total_rows_processed_count, bad_line_count))


def write_rows_to_file(filename, array):
    with open(filename, 'w') as f:
        for row in array:
            f.write(' '.join(map(str, row)) + '\n')


def print_full_ineq_lcacbxy(ineq, no_negative_coeffs=True):
    if len(ineq) == 87:
        ineq = vs.construct_full_to_NSS_h(8, 2, 4, 2).T @ ineq
    assert len(ineq) == 129
    if np.all(ineq.astype('int') == ineq):
        ineq = ineq.astype('int')

    if no_negative_coeffs:
        for a1, a2, c, b, x1, x2, y in itertools.product(B, repeat=7):
            coeff = ineq[concatenate_bits(a1, a2, c, b, x1, x2, y)]
            if coeff < 0:
                ineq[concatenate_bits(a1, a2, c, b, x1, x2, y)] = 0
                # Add -coeff to the RHS of the inequality and -coeff*(1-p(a1,a2,c,b|x1,x2,y)) to the LHS
                ineq[-1] += coeff  # not -=, because ineq[-1] is the negated bound
                for _a1, _a2, _c, _b in itertools.product(B, repeat=4):
                    if (_a1, _a2, _c, _b) != (a1, a2, c, b):
                        ineq[concatenate_bits(_a1, _a2, _c, _b, x1, x2, y)] -= coeff

    string = ''
    spaces_per_term = 15
    for a1, a2, c, b in itertools.product(B, repeat=4):
        for x1, x2, y in itertools.product(B, repeat=3):
            coeff = ineq[concatenate_bits(a1, a2, c, b, x1, x2, y)]
            p = 'p(%d%d%d%d|%d%d%d)' % (a1, a2, c, b, x1, x2, y)
            if coeff == 1:
                term = '+ ' + p
            elif coeff == -1:
                term = '- ' + p
            elif coeff == 0:
                term = ''
            elif coeff > 0:
                term = '+' + str(coeff) + p
            else:
                term = str(coeff) + p
            string += term + ' ' * (spaces_per_term - len(term))
        string += '\n'
    string = string[:-1]  # remove last newline
    string += ' ≤ ' + str(-ineq[-1])
    print(string)
    return string



def construct_deterministic_cor_nss_homog(a, b, na, nb, nx, ny):
    """ a,b should be functions (e.g. lambdas) from nx×ny->na, and nx×ny->nb. The returned vector is full homogeneous. """
    cor = np.zeros((na, nb, nx, ny), dtype='int')
    for _a, _b, _x, _y in cart(range(na), range(nb), range(nx), range(ny)):
        cor[_a, _b, _x, _y] = 1 * (_a == a(_x, _y)) * (_b == b(_x, _y))
    return vs.construct_full_to_NSS_h(na, nb, nx, ny) @ np.r_[cor.reshape((na * nb * nx * ny,)), [1]]


def deterministic_cor_full_homog(a1, a2, c, b):
    """ a1, a2, c, b should be functions (e.g. lambdas) from three binary variables to one binary variable. The returned vector is full homogeneous, so of length 129. """
    cor = np.zeros((2,) * 7, dtype='int')
    for _a1, _a2, _c, _b, x1, x2, y in itertools.product((0, 1), repeat=7):
        cor[_a1, _a2, _c, _b, x1, x2, y] = 1 * (_a1 == a1(x1, x2, y)) * (_a2 == a2(x1, x2, y)) * (_c == c(x1, x2, y)) * (_b == b(x1, x2, y))
    return np.r_[cor.reshape((128,)), [1]]


def one_hot_vector(size, position, dtype='int'):
    if position < 0:
        position += size
    return np.r_[np.zeros(position, dtype=dtype), [1], np.zeros(size - position - 1, dtype=dtype)]


def reciprocal_or_zero(array):
    return np.reciprocal(array, out=np.zeros_like(array), where=array != 0)


def normalise_h(row):
    """ Rescale a vector so as to have the last coordinate equal to 1. """
    return 1/row[-1] * row


def max_violation_h(cors, facets):
    cors = np.array(cors)
    facets = np.array(facets)

    assert len(cors.shape) in [1, 2]
    if len(cors.shape) == 1:
        cors = cors.reshape((1, len(cors)))

    assert len(facets.shape) in [1, 2]
    if len(facets.shape) == 1:
        facets = facets.reshape((len(cors), 1))

    # renormalise the cors
    for i in range(len(cors)):
        cors[i] = 1. / cors[i][-1] * cors[i]

    return np.max(facets @ cors.T)


B = (0,1)

def cart(*args):
    return list(itertools.product(*args))


def concatenate_bits(*bits):
    # Example: concatenate_bits(1,0,1) returns 5
    return sum([bits[i] << (len(bits) - i - 1) for i in range(0, len(bits))])
