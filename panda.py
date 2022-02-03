import itertools
import numpy as np
import polytope_utils

from symmetry_utils import nss_symmetry_generators

## NSS (i.e. pure Bell) stuff
def nss_write_panda_input(na, nb, nx, ny, readable=False, filename=None):
    """ See http://comopt.ifi.uni-heidelberg.de/software/PANDA/format.html for format of PANDA input file.
    :param filename: if None, then automatically generated.
    :param readable: if True, use human-readable variable names (which are however longer). """
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    dim_NSS = polytope_utils.dim_NSS(na, nb, nx, ny)
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    if readable:
        var_names = nss_readable_var_names(na, nb, nx, ny)
    else:
        var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in nss_symmetry_generators(na, nb, nx, ny, var_names):
        lines.append(symm)

    # 5) Inequalities and equations
    lines.append('Inequalities:')
    # the ineqs are -p(ab|xy) <= 0 for all a,b,x,y. For the final nx * ny rows, note that this means -(p(na-1 nb-1|xy) - 1) <= 1
    # except for the final nx * ny rows, which represent , and must therefore be >= -1.
    matrix = polytope_utils.construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny).astype('int8')
    dim_full = na * nb * nx * ny
    for i in range(0, dim_full):
        lines.append(' '.join(map(str, -matrix[i])) + (' 0' if i < dim_full - nx * ny else ' -1'))

    # Write to file
    if filename is None:
        filename = 'panda-files/nss_facets_%d-%d-%d-%d.pi' % (na, nb, nx, ny)
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()

    # This works for 2,2,2,2! Running this through PANDA gives exacty the two equivalent classes we expect from BLM+05 Eqs. (8) and (9).
    # Namely:
    #   1  1  1  1  1  1  1  1  1  -> the local, deterministic vertex
    #   1  1  1  1  1  1  1  0  2  -> the nonlocal, nondeterministic vertex (I checked that it's indeed the same as Eqs. (8) and (9).


def nss_readable_var_names(na, nb, nx, ny):
    var_names = []
    # NSS-I
    for a, x in itertools.product(range(0, na - 1), range(0, nx)):
        var_names.append('a' + str(a) + 'x' + str(x))
    # NSS-II
    for b, y in itertools.product(range(0, nb - 1), range(0, ny)):
        var_names.append('b' + str(b) + 'y' + str(y))
    # NSS-III
    for a, b, x, y in itertools.product(range(0, na - 1), range(0, nb - 1), range(0, nx), range(0, ny)):
        var_names.append('a' + str(a) + 'b' + str(b) + 'x' + str(x) + 'y' + str(y))
    return var_names


def nss_vertex_classes_from_BLM05(na, nb):
    """ Calculated from BLM+05 Eq. (12), which assumes nx = ny = 2. """
    result = []
    full_to_NSS_matrix = polytope_utils.construct_full_to_NSS_matrix(na, nb, 2, 2)
    # Literally follow BLM+05 Eq. (12)
    for k in range(1, min(na, nb) + 1):
        # First calculate full-dim cor
        cor = np.zeros(na * nb * 2 * 2)
        i = 0
        for a, b, x, y in itertools.product(range(0, na), range(0, nb), (0, 1), (0, 1)):
            # k = 1 is the local vertex; k > 1 are the nonlocal vertices.
            if (k == 1 and (a, b) == (0, 0)) \
                    or (k != 1 and a < k and b < k and (b - a) % k == x * y):  # The nonlocal vertices
                cor[i] = 1  # Division by k is indicated by adding a k in on the end of the NSS vector - just like PANDA output files.
            i += 1
        result.append(np.r_[full_to_NSS_matrix @ cor, [k]])

    # Some formatting, comment out if not desired
    result = np.array(result, dtype='int8')  # convert to integers
    result = ['  ' + '  '.join(map(str, row)) for row in result]  # format each vertex class
    result = '\n'.join(result)
    return result
    # Using this, I checked the PANDA output for the 3,3,2,2 case. Namely
    # panda_vertex = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 0, 0]
    # BLM05_vertex = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0]
    # print(nss_vertices_are_equivalent(3, 3, 2, 2, panda_vertex, BLM05_vertex, 5))
    # Gives True, so the vertices are equivalent.


## NSCO1 stuff
# ...