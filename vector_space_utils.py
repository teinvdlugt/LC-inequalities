import functools
import itertools
import math
import sys
import time

import numpy as np
import numpy.linalg

import panda
import symmetry_utils
import towards_lc
import utils
import vector_space_utils

B = (0, 1)


## NSS stuff
def dim_NSS(na, nb, nx, ny):
    return nx * ny * (na - 1) * (nb - 1) + nx * (na - 1) + ny * (nb - 1)


def construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny):
    """ Returns matrix of shape (nx*ny*na*nb, nx*ny*(na-1)(nb-1)+nx(na-1)+ny(nb-1)) that converts NSS coords
        {p(a|x), p(b|y), p(ab|xy)  |  a<na-1, b<nb-1, all x,y} =: NSS-I ∪ NSS-II ∪ NSS-III (see [p113])
    into 'weird' full coords
        {p(ab|xy)  |  (a,b) != (na-1,nb-1), all x,y} ∪ {p(na,nb|xy) - 1  |  all x,y}.
    NOTE This function assumes nx,ny,na,nb are all powers of 2, so that the coordinates can be labelled in the usual way. tODO maybe it doesn't
    (specifically, this assumption is used in using concatenate_bits)
    """
    # Define dimensions
    full_dim = na * nb * nx * ny
    NSS_dim = dim_NSS(na, nb, nx, ny)
    matrix = np.zeros((full_dim, NSS_dim), dtype='int')

    # Define ranges of variables
    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)

    # Useful functions to work with NSS coords
    NSS_II_offset = (na - 1) * nx  # the index at which NSS coords of type II [see p113] start within a NSS vector
    NSS_III_offset = NSS_II_offset + (nb - 1) * ny  # the index at which NSS coords of type III [see p113] start within a NSS vector
    get_NSS_I_index = lambda a, x: a * nx + x
    get_NSS_II_index = lambda b, y: NSS_II_offset + b * ny + y
    get_NSS_III_index = lambda a, b, x, y: NSS_III_offset + a * (nb - 1) * nx * ny + b * nx * ny + x * ny + y  # b only ranges over 0,...,nb-2

    # Loop over a,b,x,y, defining each p(ab|xy) in each iteration.
    current_row = 0
    for a, b, x, y in itertools.product(ra, rb, rx, ry):
        # See [p113] in Sketchbook for expressions in terms of NSS coords
        if a != na - 1 and b != nb - 1:
            # p(ab|xy) appears in NSS itself
            matrix[current_row][get_NSS_III_index(a, b, x, y)] = 1
        elif a == na - 1 and b != nb - 1:
            # p(na-1 b|xy) = p(b|y) - sum_{_a < na-1} p(_a b|xy)
            matrix[current_row][get_NSS_II_index(b, y)] = 1
            for _a in range(0, na - 1):
                matrix[current_row][get_NSS_III_index(_a, b, x, y)] = -1
        elif a != na - 1 and b == nb - 1:
            # p(a nb-1|xy) = p(a|x) - sum_{_b < nb-1} p(a _b|xy)
            matrix[current_row][get_NSS_I_index(a, x)] = 1
            for _b in range(0, nb - 1):
                matrix[current_row][get_NSS_III_index(a, _b, x, y)] = -1
        elif a == na - 1 and b == nb - 1:
            # (For the current values of x,y), the rows corresponding to p(ab|xy) with (a,b)!=(na-1,nb-1) have already been filled.
            # So can use those to calculate the current row, which is -sum_{(a,b)!=(na-1,nb-1)} p(ab|xy)
            # NOTE that this isn't p(na-1,nb-1|xy) but instead it is p(na-1,nb-1|xy) - 1.
            for _a, _b in list(itertools.product(ra, rb))[0: na * nb - 1]:  # everything except (na-1, nb-1)
                matrix[current_row] -= matrix[_a * nb * nx * ny + _b * nx * ny + x * ny + y]
        else:
            print("something went wrong")
        current_row += 1
    return matrix


def construct_NSS_to_full_homogeneous(na=8, nb=2, nx=4, ny=2):
    return np.block([
        [construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny), beta(na * nb * nx * ny, nx * ny).reshape((na * nb * nx * ny, 1)).astype('int')],
        [np.zeros(dim_NSS(na, nb, nx, ny), dtype='int'), 1]
    ])


def construct_full_to_NSS_matrix(na, nb, nx, ny):
    """
    Make and return matrix that converts shape (na*nb*nx*ny) vectors (full dimension) to shape (dim_NSS(na, nb, nx, ny),)
    vectors (NSS representation). ith row of the matrix represents ith NSS coordinate
    """

    full_dim = na * nb * nx * ny
    NSS_dim = dim_NSS(na, nb, nx, ny)
    matrix = np.zeros((NSS_dim, full_dim), dtype='int8')

    # Define ranges of variables in NSS rep
    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)

    get_index_in_full_rep = lambda a, b, x, y: a * nb * nx * ny + b * nx * ny + x * ny + y

    current_row = 0

    # NSS-I: p(a|x)
    for a, x in itertools.product(range(0, na - 1), rx):
        # p(a|x) = sum_b p(ab|x0)
        for b in rb:
            matrix[current_row][get_index_in_full_rep(a, b, x, 0)] = 1
        current_row += 1

    # NSS-II: p(b|y)
    for b, y in itertools.product(range(0, nb - 1), ry):
        # p(b|y) = sum_a p(ab|0y)
        for a in ra:
            matrix[current_row][get_index_in_full_rep(a, b, 0, y)] = 1
        current_row += 1

    # NS-III: p(ab|xy)
    for a, b, x, y in itertools.product(range(0, na - 1), range(0, nb - 1), rx, ry):
        matrix[current_row][get_index_in_full_rep(a, b, x, y)] = 1
        current_row += 1

    return matrix


def construct_full_to_NSS_homog(na, nb, nx, ny):
    return np.block([[construct_full_to_NSS_matrix(na, nb, nx, ny), np.zeros((dim_NSS(na, nb, nx, ny), 1), dtype='int')],
                     [np.zeros(na * nb * nx * ny, dtype='int'), 1]])


def full_acb_to_nss_homog(cor_full, common_multiple_of_denominators=None):
    """ If common_multiple_of_denominators is not None, the vector is approximated by an integer one. """
    assert vector_space_utils.is_in_NSS(cor_full, 8, 2, 4, 2)
    cor_nss = construct_full_to_NSS_matrix(8, 2, 4, 2) @ cor_full
    assert np.all(cor_nss <= 1)
    if common_multiple_of_denominators is not None:
        cor_nss_rescaled = common_multiple_of_denominators * cor_nss
        cor_nss_rescaled_approx = utils.approximate(cor_nss_rescaled, [n for n in range(common_multiple_of_denominators + 1)])
        cor_nss_approx_homog = np.r_[cor_nss_rescaled_approx, [common_multiple_of_denominators]]
        cor_nss_approx_homog_int = cor_nss_approx_homog.astype('int64')
        assert np.all(cor_nss_approx_homog == cor_nss_approx_homog_int)
        gcd = functools.reduce(math.gcd, cor_nss_approx_homog_int)
        cor_nss_approx_homog_normalised = (1 / gcd) * cor_nss_approx_homog
        cor_nss_approx_homog_normalised_int = cor_nss_approx_homog_normalised.astype('int64')
        if not np.all(cor_nss_approx_homog_normalised == cor_nss_approx_homog_normalised_int):
            print("Warning: your value of common_multiple_of_denominators was not correct")  # wrong?
        return cor_nss_approx_homog_normalised_int
    else:
        return np.r_[cor_nss, [1]]


def is_in_NSS(cor, na, nb, nx, ny, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length na*nb*nx*ny (or homogeneous version, len+1, also works) """
    if len(cor) == na * nb * nx * ny + 1:
        assert cor[-1] != 0
        cor = cor[:-1]  # actually doing the rescaling is unnecessary as it does not impact signalling
    else:
        assert len(cor) == na * nb * nx * ny

    # Probabilities >= 0
    if np.any(cor < -tol):
        return False

    cor_ab_xy = cor.reshape((na, nb, nx, ny))
    cor_b_xy = np.einsum('ijkl->jkl', cor_ab_xy)
    cor_a_xy = np.einsum('ijkl->ikl', cor_ab_xy)

    # Check for signalling A -> B
    for b, y in cart(range(0, nb), range(0, ny)):
        pby = cor_b_xy[b, 0, y]
        for x in range(1, nx):
            if abs(pby - cor_b_xy[b, x, y]) > tol:
                return False
    for a, x in cart(range(0, na), range(0, nx)):
        pax = cor_a_xy[a, x, 0]
        for y in range(1, ny):
            if abs(pax - cor_a_xy[a, x, y]) > tol:
                return False
    return True


def is_in_NSCO1(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    # First check if in NSS. This also checks if all probs are >=0.
    if not is_in_NSS(cor, 8, 2, 4, 2, tol):
        return False

    # Left to check: a1b independent of x2
    cor = cor.reshape((2,) * 7)
    cor_a1b_x1x2y = np.einsum('ijklmno->ilmno', cor)
    for a1, b, x1, y in itertools.product((0, 1), repeat=4):
        if abs(cor_a1b_x1x2y[a1, b, x1, 0, y] - cor_a1b_x1x2y[a1, b, x1, 1, y]) > tol:
            return False
    return True


def is_in_NSCO1st(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    # First check if in NSS. This also checks if all probs are >=0.
    if not is_in_NSS(cor, 8, 2, 4, 2, tol):
        return False

    # Left to check: a1 independent of x2
    cor = cor.reshape((2,) * 7)
    cor_a1_x1x2 = np.einsum('ijklmno->imno', cor)[:, :, :, 0]
    for a1, x1 in itertools.product((0, 1), repeat=2):
        if abs(cor_a1_x1x2[a1, x1, 0] - cor_a1_x1x2[a1, x1, 1]) > tol:
            return False
    return True


def is_in_NSCO2(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
    cor_swapped = swap_A1_A2_matrix @ cor
    return is_in_NSCO1(cor_swapped, tol)


def is_in_NSCO2st(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
    cor_swapped = swap_A1_A2_matrix @ cor
    return is_in_NSCO1st(cor_swapped, tol)


def generate_dictionary_of_full_probs_in_NSS_coords(filename=None):
    with open(filename, 'w') as f:
        NtoF = construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2).astype('int8')
        for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
            i = concatenate_bits(a1, a2, c, b, x1, x2, y)
            f.write('p(%d%d%d%d|%d%d%d): ' % (a1, a2, c, b, x1, x2, y) + ' '.join(map(str, -NtoF[i])) + (' 0' if i < 120 else ' -1') + '\n')


def write_cor_to_file(cor, filename):
    with open(filename, 'w') as f:
        f.write('p(a1 a2 c b | x1 x2 y)\n')
        for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
            i = concatenate_bits(a1, a2, c, b, x1, x2, y)
            f.write('p(%d%d%d%d|%d%d%d): %s' % (a1, a2, c, b, x1, x2, y, str(cor[i])) + '\n')


## NSCO1 stuff (strong version, i.e. dim=80).
def dim_NSCO1(nabcxy):
    na1, na2, nc, nb, nx1, nx2, ny = nabcxy
    # For weak NSCO1 version: nx1 * (na1 - 1) + nx1 * nx2 * na1 * (na2 * nc - 1) + ny * (nb - 1) + nx1 * nx2 * ny * (na1 * na2 * nc - 1) * (nb - 1)
    # See [p131]:
    return nx1 * (na1 - 1) + ny * (nb - 1) + nx1 * ny * (na1 - 1) * (nb - 1) + nx1 * nx2 * na1 * (na2 * nc - 1) + nx1 * nx2 * ny * na1 * (na2 * nc - 1) * (nb - 1)


def dim_full_abcxy(nabcxy):
    assert len(nabcxy) == 7
    return nabcxy[0] * nabcxy[1] * nabcxy[2] * nabcxy[3] * nabcxy[4] * nabcxy[5] * nabcxy[6]


def construct_full_to_NSCO1_matrix():
    """
    Make and return matrix that converts length-128 vectors to length-80 vectors (NSCO1 rep of [p131])
    NOTE this function assumes the order first c, then b.
    """

    matrix = np.zeros((80, 128))

    get_index_in_full_rep = lambda a1, a2, c, b, x1, x2, y: concatenate_bits(a1, a2, c, b, x1, x2, y)

    current_row = 0

    # NSCO1-I: p(a1=0|x1)
    for x1 in B:
        for a2, b, c in cart(B, B, B):
            matrix[current_row][get_index_in_full_rep(0, a2, c, b, x1, 0, 0)] = 1
        current_row += 1
    # NSCO1-II: p(b=0|y)
    for y in B:
        for a1, a2, c in cart(B, B, B):
            matrix[current_row][get_index_in_full_rep(a1, a2, c, 0, 0, 0, y)] = 1
        current_row += 1
    # NSCO1-III: p(a1=0 b=0|x1,y)
    for x1, y in cart(B, B):
        for a2, c in cart(B, B):
            matrix[current_row][get_index_in_full_rep(0, a2, c, 0, x1, 0, y)] = 1
        current_row += 1
    # NSCO1-IV: p(a1,a2,c|x1,x2), (a2,c) != (1,1)
    for a1, (a2, c), x1, x2 in cart(B, cart(B, B)[:-1], B, B):
        for b in B:
            matrix[current_row][get_index_in_full_rep(a1, a2, c, b, x1, x2, 0)] = 1
        current_row += 1
    # NSCO1-V: p(a1,a2,c,0|x1,x2,y), (a2,c) != (1,1)
    for a1, (a2, c), x1, x2, y in cart(B, cart(B, B)[:-1], B, B, B):
        matrix[current_row][get_index_in_full_rep(a1, a2, c, 0, x1, x2, y)] = 1
        current_row += 1

    return matrix  # I tested this function, see __main__ below


def construct_full_to_NSCO1_homog():
    return np.block([
        [construct_full_to_NSCO1_matrix(), np.zeros((80, 1), dtype='int')],
        [np.zeros(128, dtype='int'), 1]
    ])


def construct_NSCO1_to_full_weird_matrix():
    """
    Make and return shape-(128,80) matrix that converts length-80 vectors representing the prob distr p in the
    NSCO1 representation to the length-128 vector representing p-β in the standard/'full' representation.
    NOTE this function assumes the order first c, then b.
    """

    NSCO1_II_offset = 2
    NSCO1_III_offset = NSCO1_II_offset + 2
    NSCO1_IV_offset = NSCO1_III_offset + 4
    NSCO1_V_offset = NSCO1_IV_offset + 24

    NSCO1_I_index = lambda x1: x1
    NSCO1_II_index = lambda y: NSCO1_II_offset + y
    NSCO1_III_index = lambda x1, y: NSCO1_III_offset + x1 * 2 + y
    NSCO1_IV_index = lambda a1, a2, c, x1, x2: NSCO1_IV_offset + a1 * (2 * 2 - 1) * 2 * 2 + a2 * 2 * 2 * 2 + c * 2 * 2 + x1 * 2 + x2
    NSCO1_V_index = lambda a1, a2, c, x1, x2, y: NSCO1_V_offset + a1 * (2 * 2 - 1) * 1 * 8 + a2 * 2 * 8 + c * 8 + x1 * 4 + x2 * 2 + y

    matrix = np.zeros((128, 80), dtype='int8')
    current_row = 0  # Alternatively, can use concatenate_bits(a1, a2, c, b, x1, x2, y) in each iteration of the below for-loop.

    # See [p133]
    for a1, a2, c, b, x1, x2, y in cart(B, B, B, B, B, B, B):
        if b == 0:
            if (a2, c) != (1, 1):
                # case (i)  [cf. p133]
                matrix[current_row][NSCO1_V_index(a1, a2, c, x1, x2, y)] = 1
            elif a1 == 0:
                # case (ii)
                matrix[current_row][NSCO1_III_index(x1, y)] = 1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_V_index(0, _a2, _c, x1, x2, y)] = -1
            elif a1 == 1:
                # case (iii)
                matrix[current_row][NSCO1_II_index(y)] = 1
                matrix[current_row][NSCO1_III_index(x1, y)] = -1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_V_index(1, _a2, _c, x1, x2, y)] = -1
        elif b == 1:
            if (a2, c) != (1, 1):
                # case (iv)
                matrix[current_row][NSCO1_IV_index(a1, a2, c, x1, x2)] = 1
                matrix[current_row][NSCO1_V_index(a1, a2, c, x1, x2, y)] = -1
            elif a1 == 0:
                # case (v)
                matrix[current_row][NSCO1_I_index(x1)] = 1
                matrix[current_row][NSCO1_III_index(x1, y)] = -1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_IV_index(0, _a2, _c, x1, x2)] = -1
                    matrix[current_row][NSCO1_V_index(0, _a2, _c, x1, x2, y)] = 1
            elif a1 == 1:
                # case (vi): p(1111|x1x2y)-1
                matrix[current_row][NSCO1_I_index(x1)] = -1
                matrix[current_row][NSCO1_II_index(y)] = -1
                matrix[current_row][NSCO1_III_index(x1, y)] = 1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_IV_index(1, _a2, _c, x1, x2)] = -1
                    matrix[current_row][NSCO1_V_index(1, _a2, _c, x1, x2, y)] = 1

                # Let's check that this is the same as -1 * the sum of all previous p(a1a2cb|x1x2y), for the same x1x2y.
                check_row = np.zeros(80)
                for _a1, _a2, _c, _b in cart(B, B, B, B)[:-1]:
                    check_row -= matrix[concatenate_bits(_a1, _a2, _c, _b, x1, x2, y)]
                assert np.all(check_row == matrix[current_row])

        current_row += 1
    return matrix  # I tested this function, see __main__ below


def construct_NSCO1_to_full_homogeneous():
    return np.block([
        [construct_NSCO1_to_full_weird_matrix(), beta().reshape((128, 1)).astype('int')],
        [np.zeros(80, dtype='int'), 1]
    ])


def beta(length=128, num_of_ones=8, dtype='int'):
    """ Returns the length-128 vector that has a 1 in every position corresponding to p(1111|x1x2y) for some x1,x2,y. """
    return np.r_[np.zeros(length - num_of_ones, dtype=dtype), np.ones(num_of_ones, dtype=dtype)]


## LC stuff (using 86-dim NSS parameterisation)
def construct_NSCO1_to_NSS_matrix():
    return construct_full_to_NSS_matrix(8, 2, 4, 2) @ construct_NSCO1_to_full_weird_matrix()
    # 'Weirdness' (i.e. the absence of beta()) doesn't matter, because construct_full_to_NSS_matrix will never touch the values of
    # p(1111|x1x2y); always either b != 1 or a1a2c != 111 or both.


def NSCO1_to_NSS_with_denominator(row):
    return np.r_[construct_NSCO1_to_NSS_matrix() @ np.array(row[:-1]), [row[-1]]]


## GENERAL UTILS
def are_vectors_lin_indep(vector_list):
    # Can use the following methods:
    #  - reduce to reduced row echelon form and check if last column is nonzero.
    #       - sympy.Matrix().rref https://stackoverflow.com/questions/15638650/is-there-a-standard-solution-for-gauss-elimination-in-python
    #       - scipy.linalg.lu. pl, u = lu(a, permute_l=True). https://stackoverflow.com/questions/15638650/is-there-a-standard-solution-for-gauss-elimination-in-python
    #  - compute rank of matrix which has the vectors as rows. numpy.linalg.matrix_rank
    # Use all methods, just because I'm dumb

    matrix = np.array(vector_list)

    # numpy matrix_rank:
    answer1 = numpy.linalg.matrix_rank(matrix) == len(vector_list)

    # scipy.linalg.lu:
    # answer2 = not np.all(scipy.linalg.lu(matrix, permute_l=True)[-1][-1] == 0)

    # sympy.Matirx().rref():
    # answer3 = not np.all(sympy.Matrix(matrix).rref()[-1] == 0)

    return answer1  # and answer2 and answer3


def concatenate_bits(*bits):
    # Example: concatenate_bits(1,0,1) returns 5
    return sum([bits[i] << (len(bits) - i - 1) for i in range(0, len(bits))])


def reduce_to_lin_indep_subset(matrix: numpy.ndarray, limit=None, print_progress=True):
    """ Loops through rows of matrix from front to back and removes those rows that are in the linear span of the preceding ones.
     Function leaves the passed argument unchanged, but returnes the reduced matrix.
     It stops once it has found `max_dimension` independent rows, if max_dimension is not None. """
    i = 2
    while i < len(matrix) + 1:
        if numpy.linalg.matrix_rank(matrix[0:i]) < i:
            # The first i rows of matrix are linearly dependent. So remove row i-1.
            matrix = numpy.delete(matrix, i - 1, 0)
        else:
            # The first i rows of matrix are linearly independent.
            if i == limit:
                if print_progress:
                    print("Found %d linearly independent rows, max_dimension reached; %d rows left unchecked" % (i, len(matrix) - i))
                return matrix[0:i]
            i += 1

        if print_progress:
            print("Found %d linearly independent rows, %d rows left to check" % (i - 1, len(matrix) - i + 1), end='\r')
            sys.stdout.flush()

    print()  # to get rid of the last \r
    return matrix


def reduce_file_to_lin_indep_subset(filename, lcm_of_denominators, max_dimension=None, constraint=None, print_progress=True, output_filename=None):
    """ Reads vertices from a file (specified by filename), waiting for user confirmation every time it encounters an empty line.
        Outputs a subset of linearly independent rows, stopping when max_dimension is reached (if not None).
        `constraint` is a function which takes a 'row-with-denominator' list as input and returns a boolean. If False, this
        particular row in the input file is ignored.
    """
    file = open(filename, 'r')
    batch_count = 0  # Every chunk of lines in the file separated by witregels is called a batch here
    matrix = np.zeros((0, 86), dtype='int8')

    line = file.readline()
    while line:
        # Scan for next batch, i.e. skip empty lines
        while line and (not line.strip()):
            line = file.readline()

        # Read next batch
        current_batch = []
        batch_size_unconstrained = 0
        while line.strip():  # while line is not just whitespace
            row = np.array(list(map(int, line.split())), dtype='int8')
            batch_size_unconstrained += 1
            if (constraint is None) or constraint(row):
                assert int(lcm_of_denominators / row[-1]) == lcm_of_denominators / row[-1]  # Check whether lcm_of_denominators param is correct
                scaled_row = int(lcm_of_denominators / row[-1]) * row[:-1]
                current_batch.append(scaled_row)
            line = file.readline()

        # Batch is loaded; now process it
        batch_count += 1
        print("Processing batch #%d of size %d (%d)" % (batch_count, len(current_batch), batch_size_unconstrained))
        if len(current_batch) == 0:
            print("Batch is empty")
            continue

        matrix = reduce_to_lin_indep_subset(np.r_[matrix, np.array(current_batch, dtype='int8')], max_dimension, print_progress)

        # Print progress update and save progress
        print("Processed batch #%d; %d linearly independent rows across all processed batches" % (batch_count, len(matrix)))
        if output_filename:
            output_file = open(output_filename, 'w')
            for vec in matrix:
                output_file.write(panda.row_with_denom_to_vector_str(np.r_[vec, [lcm_of_denominators]]) + "\n")
            output_file.close()
        if len(matrix) >= max_dimension:
            print("Limit reached!")
            return
        if not line:
            print("End of input file reached")
            return


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


def cart(*args):
    return list(itertools.product(*args))


if __name__ == '__main__':
    ## To test whether construct_full_to_NSCO1_matrix and construct_NSCO1_to_full_weird_matrix are correct:
    """
    FtoN = construct_full_to_NSCO1_matrix()
    NtoF = construct_NSCO1_to_full_weird_matrix()
    # For any length-128 vector f satisying the linear NSCO1 constraints, we should have  f = NtoF @ FtoN @ f + beta();
    # For any length-80 vector n, we should have  n = FtoN(NtoF(n) + beta) = FtoN @ NtoF @ n + FtoN @ beta().
    # Test it on a correlation that we know is NSCO1:
    cor128 = one_switch_4mmts.qm_corr_one_switch_4mmts_3stngs(
        rho_ctb=qm.proj(qm.kron(qm.ket0, qm.phi_plus)),
        X1=[qm.random_real_onb(), qm.random_real_onb()],
        X2=[qm.random_real_onb(), qm.random_real_onb()],
        Y=[qm.random_real_onb(), qm.random_real_onb()],
        c_onb=qm.random_real_onb()).reshape((2,) * 7).swapaxes(2, 3).reshape((128,))
    print(np.where(utils.not_almost_equal(cor128, NtoF @ FtoN @ cor128 + beta(), tol=1e-15)))
    # NOTE It returns [115,...,127] when tol=1e-16. Apparently the error blows up to above 1e-16 specifically for the last 13 indices -
    # NOTE  that makes sense, because the expression of those quantities in terms of NSCO1 parameters involves many terms, to much add-up of error.
    cor80 = np.random.rand(80)
    print(np.where(utils.not_almost_equal(cor80, FtoN @ NtoF @ cor80 + FtoN @ beta())))
    """

    ## To test NSCO1 to NSS parameterisation conversion functions:
    """
    row = list(map(int,
                   "2  2  1  1  1  1  1  1  2  1  1  1  0  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2".split()))
    vector_NSCO1 = panda.row_with_denom_to_vector(row)
    vector_NSS = construct_NSCO1_to_NSS_matrix() @ vector_NSCO1
    assert vector_NSCO1[2] == vector_NSS[28]  # p(0|y) == p(0|y). Cf. [p131, p158]
    assert vector_NSCO1[2 + 2 + 4 + 24 + 3 * 8 + 2 * 8 + 1] == vector_NSS[30 + 0b110001]  # p(1100|001) == p(1100|001)
    assert vector_NSS[0b01100] == vector_NSCO1[0] - vector_NSCO1[8] - vector_NSCO1[8 + 4] - vector_NSCO1[8 + 2 * 4]  # p(a1=0 a2=1 c=1|x1=0 x2=0) == p(a1=0|0) - p(000|00) - p(001|00) - p(010|00)
    print('  '.join(map(str, vector_NSS)))
    print('    '.join(map(str, NSCO1_to_NSS_with_denominator(row))))
    """

    ## To test reduce_to_lin_indep_subset():
    """
    matrix = np.array([[1, 1, 0, 1],
                       [2, 2, 0, 2],
                       [1, 2, 0, 1],
                       [2, 3, 0, 2],
                       [0, 1, 1, 0],
                       [1, 0, 0, 1]])
    print(reduce_to_lin_indep_subset(matrix))
    """
