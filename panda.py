import itertools

import numpy as np

import symmetry_utils
import utils
from utils import cart, B
import vector_space_utils as vs


# PANDA SYNTAX

def write_panda_input_file(dimension, var_names=None, symmetries=None, inequalities=None, vertices=None, reduced=False, filename=None):
    """ General utility function to write PANDA input files. See http://comopt.ifi.uni-heidelberg.de/software/PANDA/format.html for format of PANDA input files. """
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    lines.append('DIM=%d' % dimension)

    # 2) Names of coordinates
    lines.append('Names:')
    if var_names is None:
        var_names = ['x' + str(i) for i in range(0, dimension)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    if symmetries is not None:
        lines.append('Maps:')
        for symm in symmetries:
            lines.append(symmetry_utils.symm_matrix_to_string(symm, var_names))

    # 5) Inequalities and equations
    if (inequalities is None and vertices is None) or (inequalities is not None and vertices is not None):
        raise ValueError('Exactly one of inequalities and vertices must be None.')
    if inequalities is not None:
        lines.append('Reduced Inequalities:' if reduced else 'Inequalities:')
        for ineq in inequalities:
            lines.append(' '.join(map(str, ineq)))
    elif vertices is not None:
        lines.append('Reduced Vertices:' if reduced else 'Vertices:')
        for vert in vertices:
            lines.append(homog_vertex_to_str_with_fractions(vert))

    # Write to file, or print to screen
    if filename is None:
        print('\n'.join([line for line in lines]))
    else:
        file = open(filename, 'w')
        file.write('\n'.join(lines))
        file.close()


def homog_vertex_to_str_with_fractions(row):
    if not np.all(row == np.array(row, dtype='int')):
        raise ValueError('This function only makes sense if the passed vector has only integer values')
    from fractions import Fraction
    vector = list(row[:-1])
    denominator = row[-1]
    for i in range(len(vector)):
        vector[i] = str(Fraction(vector[i], denominator))
    return ' '.join(vector)


def convert_panda_output_vertex_to_input_vertex_format(old_filename, new_filename):
    old_file = open(old_filename, 'r')
    new_lines = []
    for line in old_file.readlines():
        try:
            new_lines.append(homog_vertex_to_str_with_fractions(list(map(int, line.split()))))
        except ValueError:
            # Line was not filled with integers. Probably a comment/header line. Leave intact but remove the newline.
            new_lines.append(line.strip())
            pass
    old_file.close()
    new_file = open(new_filename, 'w')
    new_file.write('\n'.join(new_lines))
    new_file.close()


# VAR NAMES

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


def nss_readable_var_names_a1a2c():
    var_names = []
    # NSS-I
    for (a1, a2, c), x1, x2 in cart(cart((0, 1), (0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        var_names.append('a' + str(a1) + str(a2) + 'c' + str(c) + 'x' + str(x1) + str(x2))
    # NSS-II
    for b, y in itertools.product((0,), (0, 1)):
        var_names.append('b' + str(b) + 'y' + str(y))
    # NSS-III
    for (a1, a2, c), b, x1, x2, y in cart(cart((0, 1), (0, 1), (0, 1))[:-1], (0,), (0, 1), (0, 1), (0, 1)):
        var_names.append('a' + str(a1) + str(a2) + 'c' + str(c) + 'b' + str(b) + 'x' + str(x1) + str(x2) + 'y' + str(y))
    return var_names


def nsco1_readable_var_names():
    var_names = []

    # NSCO1-I
    for a1, x1 in vs.cart((0,), B):
        var_names.append('I' + str(a1) + str(x1))
    # NSCO1-II
    for b, y in vs.cart((0,), B):
        var_names.append('II' + str(b) + str(y))
    # NSCO1-III
    for x1, y in vs.cart(B, B):
        var_names.append('III00' + str(x1) + str(y))
    # NSCO1-IV
    for a1, (a2, c), x1, x2 in vs.cart(B, vs.cart(B, B)[:-1], B, B):
        var_names.append('IV' + str(a1) + str(a2) + str(c) + str(x1) + str(x2))
    # NSCO1-V
    for a1, (a2, c), b, x1, x2, y in vs.cart(B, vs.cart(B, B)[:-1], (0,), B, B, B):
        var_names.append('V' + str(a1) + str(a2) + str(c) + str(b) + str(x1) + str(x2) + str(y))

    return var_names



# LC* stuff
def lco1st_H_symms_write_panda_input(readable=False, filename=None):
    """ Like nsco1_write_panda_input, but now for the 'weak version' LCO1*. This time, I won't use a LCO1* parameterisation (which would be 84-dim), but
    instead will work in NSS parameterisation and will provide PANDA with 'Equations' to specify the subspace LCO1*. The inequalities are then
    the inequalities of NSS, i.e. the positivity inequalities.
    :param filename: if None, then automatically generated.
    :param readable: if True, use human-readable variable names (which are however longer). """
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    dim_NSS = 86
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    if readable:
        var_names = nss_readable_var_names(8, 2, 4, 2)
    else:
        var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information. The symmetries of LCO1* are the same as those of NSCO1, but now we want to express them in NSS rather than NSCO1 coords.
    lines.append('Maps:')
    for perm in [
        lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, (x1 + 1) % 2, x2, y),  # x1 -> x1 + 1
        lambda a1, a2, c, b, x1, x2, y: ((a1 + x1) % 2, a2, c, b, x1, x2, y),  # a1 -> a1 + x1
        lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, (x2 + 1) % 2, y),  # x2 -> x2 + 1
        # NOTE x2 + x1*a1 would be a symmetry of LCO1* but not of NSS, and we're working in NSS coords so not allowed
        lambda a1, a2, c, b, x1, x2, y: (a1, (a2 + x1 * a1 * x2) % 2, c, b, x1, x2, y),  # a2 -> a2 + x1*a1*x2
        lambda a1, a2, c, b, x1, x2, y: (a1, a2, (c + x1 * a1 * x2 * a2) % 2, b, x1, x2, y),  # c -> c + x1*a1*x2*a2
        lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, x2, (y + 1) % 2),  # y  -> y  + 1
        lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, (b + y) % 2, x1, x2, y)  # b  -> b  + y
    ]:
        lines.append(symmetry_utils.symm_matrix_to_string(symmetry_utils.nss_var_perm_to_symm(perm), var_names))

    # 4) Equations
    lines.append('Equations:')
    # "for all x1: p(a1=0 | x1 0) - p(a1=0 | x1 1) = 0" (result for a1=1 follows from prob sum = 1)
    NtoF = vs.construct_NSS_to_full_h(8, 2, 4, 2).astype('int')
    for a1, x1 in vs.cart((0,), (0, 1)):
        vector_nss = np.zeros((dim_NSS,), dtype='int')
        # need to sum over a2, b, c and set y to 0
        for a2, b, c, y in vs.cart((0, 1), (0, 1), (0, 1), (0,)):
            vector_nss += NtoF[vs.concatenate_bits(a1, a2, c, b, x1, 0, y)][:-1]
            vector_nss -= NtoF[vs.concatenate_bits(a1, a2, c, b, x1, 1, y)][:-1]
            # We don't need to 'correct for weirdness' bc the weirdness in the two lines above (p(a1 | x1 0) and -p(a1 | x1 1)) cancels.
            # Otherwise would have had to do
            # vector_nss[:1] += (a1, a2, c, b) == (1, 1, 1, 1) and
            # vector_nss[:1] -= (a1, a2, c, b) == (1, 1, 1, 1)
        lines.append(' '.join(map(str, vector_nss)) + ' 0')

    # 5) Inequalities
    lines.append('Inequalities:')
    # the ineqs are -p(a1a2cb|x1x2y) <= 0 for all a1,a2,c,b,x1,x2,y. For the final 8 rows, note that this means -(p(1111|x1x2y) - 1) - 1 <= 0.
    # TODO could clean this up by using homogeneous coords in all relevant functions
    NtoF = vs.construct_NSS_to_full_h(8, 2, 4, 2).astype('int8')
    dim_full = 128
    for i in range(0, dim_full):
        lines.append(' '.join(map(str, -NtoF[i])))

    # Write to file
    if filename is None:
        filename = 'panda-files/arc-output/job31/tmp.pi'
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


# Bell and NSS 'no-(superluminal-)signalling' stuff
def nss_write_panda_input(na, nb, nx, ny, readable=False, filename=None):
    """ Write no-signalling polytope facets to PANDA file. """
    write_panda_input_file(dimension=vs.dim_NSS(na, nb, nx, ny),
                           var_names=nss_readable_var_names(na, nb, nx, ny) if readable else None,
                           symmetries=None,  # TODO write new symmetries function
                           inequalities=-vs.construct_NSS_to_full_h(na, nb, nx, ny)[:-1],
                           # minus sign because PANDA default is <= rather than >=. [:-1] because last row of homogeneous matrix is irrelevant.
                           reduced=False,
                           filename=filename)
    # This works for 2,2,2,2: Running this through PANDA gives exacty the two equivalent classes we expect from BLM+05 Eqs. (8) and (9).
    # Namely:
    #   1  1  1  1  1  1  1  1  1  -> the local, deterministic vertex
    #   1  1  1  1  1  1  1  0  2  -> the nonlocal, nondeterministic vertex (I checked that it's indeed the same as Eqs. (8) and (9).


# NOTE the symmetries in this function are probably not complete
def bell_write_panda_input(na, nb, nx, ny, reduced_vertices=True, readable=False, filename=None):
    """ Writes vertices of the Bell-local polytope to a PANDA file. """
    lines = []

    # 1) Dimension information
    dim_NSS = vs.dim_NSS(na, nb, nx, ny)
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    var_names = ['x' + str(i) for i in range(0, dim_NSS)] if not readable else nss_readable_var_names(na, nb, nx, ny)
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    def neighbourly_two_cycle(n, N):
        assert 0 <= n <= N - 2
        return lambda a: a + 1 if a == n else (a - 1 if a == n + 1 else a)
    def neighbourly_two_cycles(N):
        return [neighbourly_two_cycle(n, N) for n in range(0, N - 1)]

    symms = []
    for cycle in neighbourly_two_cycles(nx):
        symms.append(symmetry_utils.nss_var_perm_to_symm_more_general(lambda a, b, x, y: (a, b, cycle(x), y), na, nb, nx, ny))
    for cycle in neighbourly_two_cycles(ny):
        symms.append(symmetry_utils.nss_var_perm_to_symm_more_general(lambda a, b, x, y: (a, b, x, cycle(y)), na, nb, nx, ny))
    for _x in range(0, nx):
        for cycle in neighbourly_two_cycles(na):
            symms.append(symmetry_utils.nss_var_perm_to_symm_more_general(lambda a, b, x, y: (cycle(a) if x == _x else a, b, x, y), na, nb, nx, ny))
    for _y in range(0, ny):
        for cycle in neighbourly_two_cycles(ny):
            symms.append(symmetry_utils.nss_var_perm_to_symm_more_general(lambda a, b, x, y: (a, cycle(b) if y == _y else b, x, y), na, nb, nx, ny))

    lines.append('Maps:')
    for symm in symms:
        lines.append(symmetry_utils.symm_matrix_to_string(symm, var_names))

    # 4) Vertices
    if reduced_vertices:
        lines.append('Reduced Vertices:')
        vertex = utils.construct_deterministic_cor_nss_homog(lambda x, y: 0, lambda x, y: 0, na, nb, nx, ny)
        lines.append(' '.join(map(str, vertex[:-1])))
    else:
        lines.append('Vertices:')
        # Loop through deterministic functions nx->na and ny->nb
        for a_s in itertools.product(range(na), repeat=nx):
            for b_s in itertools.product(range(nb), repeat=ny):
                lines.append(' '.join(map(str, utils.construct_deterministic_cor_nss_homog(lambda x, y: a_s[x], lambda x, y: b_s[y], na, nb, nx, ny)[:-1])))

    # Write to file
    if filename is None:
        filename = 'panda-files/bell_polytope/bell_%d%d%d%d_vertices' % (na, nb, nx, ny)
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


def nss_vertex_classes_from_BLM05(na, nb):
    """ Calculated from Barrett et al. 2005 Eq. (12), which assumes nx = ny = 2. """
    result = []
    full_to_NSS_matrix = vs.construct_full_to_NSS(na, nb, 2, 2)
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
    # print(nss_vertices_are_equivalent_old(3, 3, 2, 2, panda_vertex, BLM05_vertex, 5))
    # Gives True, so the vertices are equivalent.
