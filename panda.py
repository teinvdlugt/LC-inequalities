import itertools
import numpy as np

import symmetry_utils
import vector_space_utils as vs

B = (0, 1)


## NSS (i.e. pure Bell) stuff
def nss_write_panda_input(na, nb, nx, ny, readable=False, filename=None):
    """ See http://comopt.ifi.uni-heidelberg.de/software/PANDA/format.html for format of PANDA input file.
    :param filename: if None, then automatically generated.
    :param readable: if True, use human-readable variable names (which are however longer). """
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    dim_NSS = vs.dim_NSS(na, nb, nx, ny)
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
    for symm in symmetry_utils.nss_symmetry_generators(na, nb, nx, ny, var_names):
        lines.append(symm)

    # 5) Inequalities and equations
    lines.append('Inequalities:')
    # the ineqs are -p(ab|xy) <= 0 for all a,b,x,y. For the final nx * ny rows, note that this means -(p(na-1 nb-1|xy) - 1) - 1 <= 0
    matrix = vs.construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny).astype('int8')
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


def bell_write_panda_input(na, nb, nx, ny, filename=None):
    lines = []

    # 1) Dimension information
    dim_NSS = vs.dim_NSS(na, nb, nx, ny)
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    # x -> perm(x) = λ + μx   (x^2 = x - 2)
    # y -> perm(y) = λ + μy   (y^2 = y)
    # b -> perm(b|y) = b + f(y) = b + λ + μy
    # a -> perm(a|x) = 2-cycle * f(x) + id * (1-f(x))
    # ^ maybe later! TODO

    # symmetry_utils.nss_var_perm_to_symm(lambda a,b,x,y: ) :(


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
    for (a1, a2, c), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        var_names.append('a' + str(a1) + str(a2) + 'c' + str(c) + 'x' + str(x1) + str(x2))
    # NSS-II
    for b, y in itertools.product((0,), (0, 1)):
        var_names.append('b' + str(b) + 'y' + str(y))
    # NSS-III
    for (a1, a2, c), b, x1, x2, y in vs.cart(vs.cart((0, 1), (0, 1), (0, 1))[:-1], (0,), (0, 1), (0, 1), (0, 1)):
        var_names.append('a' + str(a1) + str(a2) + 'c' + str(c) + 'b' + str(b) + 'x' + str(x1) + str(x2) + 'y' + str(y))
    return var_names


def nss_vertex_classes_from_BLM05(na, nb):
    """ Calculated from BLM+05 Eq. (12), which assumes nx = ny = 2. """
    result = []
    full_to_NSS_matrix = vs.construct_full_to_NSS_matrix(na, nb, 2, 2)
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


def print_full_ineq_lcacbxy(ineq, no_negative_coeffs=True):
    if len(ineq) == 87:
        ineq = vs.construct_full_to_NSS_homog(8, 2, 4, 2).T @ ineq
    assert len(ineq) == 129
    if np.all(ineq.astype('int') == ineq):
        ineq = ineq.astype('int')

    if no_negative_coeffs:
        for a1, a2, c, b, x1, x2, y in itertools.product(B, repeat=7):
            coeff = ineq[vs.concatenate_bits(a1, a2, c, b, x1, x2, y)]
            if coeff < 0:
                ineq[vs.concatenate_bits(a1, a2, c, b, x1, x2, y)] = 0
                # Add -coeff to the RHS of the inequality and -coeff*(1-p(a1,a2,c,b|x1,x2,y)) to the LHS
                ineq[-1] += coeff  # not -=, because ineq[-1] is the negated bound
                for _a1, _a2, _c, _b in itertools.product(B, repeat=4):
                    if (_a1, _a2, _c, _b) != (a1, a2, c, b):
                        ineq[vs.concatenate_bits(_a1, _a2, _c, _b, x1, x2, y)] -= coeff

    string = ''
    spaces_per_term = 15
    for a1, a2, c, b in itertools.product(B, repeat=4):
        for x1, x2, y in itertools.product(B, repeat=3):
            coeff = ineq[vs.concatenate_bits(a1, a2, c, b, x1, x2, y)]
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


## NSCO1 stuff (strong version! so 80-dim)
# NOTE Always stick to the order a1 a2 c b x1 x2 y, so c first, then b. (Esp. for NSCO1-V)

def nsco1_write_panda_input(readable=False, filename=None):
    """ See http://comopt.ifi.uni-heidelberg.de/software/PANDA/format.html for format of PANDA input file.
    :param filename: if None, then automatically generated.
    :param readable: if True, use human-readable variable names (which are however longer). """
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    dim_NSCO1 = vs.dim_NSCO1((2,) * 7)
    lines.append('DIM=%d' % dim_NSCO1)

    # 2) Names of coordinates
    lines.append('Names:')
    if readable:
        var_names = nsco1_readable_var_names()
    else:
        var_names = ['x' + str(i) for i in range(0, dim_NSCO1)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in symmetry_utils.nsco1_symm_generators():
        lines.append(symmetry_utils.symm_matrix_to_string(symm, var_names))

    # 5) Inequalities and equations
    lines.append('Inequalities:')
    # the ineqs are -p(a1a2cb|x1x2y) <= 0 for all a1,a2,c,b,x1,x2,y. For the final 8 rows, note that this means -(p(1111|x1x2y) - 1) - 1 <= 0.
    NtoF = vs.construct_NSCO1_to_full_weird_matrix().astype('int8')  # list of 'full vectors' expressed in NSCO1 vectors
    dim_full = 128
    for i in range(0, dim_full):
        lines.append(' '.join(map(str, -NtoF[i])) + (' 0' if i < 120 else ' -1'))

    # Write to file
    if filename is None:
        filename = 'panda-files/old-and-irrelevant/nsco1_facets_perm6feb.pi'
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


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


def nsco1_panda_vertex_to_full_cor(panda_vertex):
    """ Takes a string of a panda output file and returns the probability distribution that it represents as a length-128 vector. """
    splitted = panda_vertex.split()
    NSCO1_vector = np.array(splitted[:-1]).astype('float64')
    NSCO1_vector = 1 / float(splitted[-1]) * NSCO1_vector
    return vs.construct_NSCO1_to_full_weird_matrix() @ NSCO1_vector + vs.beta()


def nsco1_characterise_deterministic_vertex(vertex):
    cor = vertex.reshape((2,) * 7)

    def totuple(array):
        try:
            return tuple(totuple(i) for i in array)
        except TypeError:
            return array

    # characterise p(a1|x1)
    cor_a1_x1 = np.einsum('ijklmno->imno', cor)[:, :, 0, 0]
    print(totuple(cor_a1_x1))
    if np.all(cor_a1_x1[0] == [1, 1]):
        print('a1 = 0')
    elif np.all(cor_a1_x1[0] == [1, 0]):
        print('a1 = x1')
    elif np.all(cor_a1_x1[0] == [0, 1]):
        print('a1 = ¬x1')
    elif np.all(cor_a1_x1[0] == [1, 1]):
        print('a1 = 1')

    descriptions_dict = {
        ((1, 1), (1, 1)): 'a2 is 0',
        ((0, 0), (0, 0)): 'a2 is 1',
        ((0, 0), (1, 1)): 'a2 is x1',
        ((1, 1), (0, 0)): 'a2 is ¬x1',
        ((0, 1), (0, 1)): 'a2 is x2',
        ((1, 0), (1, 0)): 'a2 is ¬x2',
        ((1, 0), (0, 1)): 'a2 is x1+x2',
        ((0, 1), (1, 0)): 'a2 is ¬(x1+x2)'
    }

    # characterise p(a2|x1x2)
    cor_a2_x1x2 = np.einsum('ijklmno->jmno', cor)[:, :, :, 0]
    if totuple(cor_a2_x1x2) in descriptions_dict.keys():
        print(descriptions_dict[cor_a2_x1x2])
    else:
        print('cor_a2_x1x2 was not recognised:')
        print(cor_a2_x1x2)

    # characterise p(c|x1x2)
    cor_c_x1x2 = np.einsum('ijklmno->kmno', cor)[:, :, :, 0]
    if totuple(cor_c_x1x2) in descriptions_dict.keys():
        print(descriptions_dict[cor_c_x1x2])
    else:
        print('cor_c_x1x2 was not recognised:')
        print(cor_c_x1x2)

    # characterise p(b|y)
    cor_b_y = np.einsum('ijklmno->lmno', cor)[:, 0, 0, :]
    if np.all(cor_b_y[0] == [1, 1]):
        print('b = 0')
    elif np.all(cor_b_y[0] == [1, 0]):
        print('b = y')
    elif np.all(cor_b_y[0] == [0, 1]):
        print('b = ¬y')
    else:
        print('b = 1')


## LCO1* stuff (weak version)
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
    NtoF = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2).astype('int')
    for a1, x1 in vs.cart((0,), (0, 1)):
        vector_nss = np.zeros((dim_NSS,), dtype='int')
        # need to sum over a2, b, c and set y to 0
        for a2, b, c, y in vs.cart((0, 1), (0, 1), (0, 1), (0,)):
            vector_nss += NtoF[vs.concatenate_bits(a1, a2, c, b, x1, 0, y)]
            vector_nss -= NtoF[vs.concatenate_bits(a1, a2, c, b, x1, 1, y)]
            # We don't need to 'correct for weirdness' bc the weirdness in the two lines above (p(a1 | x1 0) and -p(a1 | x1 1)) cancels.
            # Otherwise would have had to do
            # vector_nss[:1] += (a1, a2, c, b) == (1, 1, 1, 1) and
            # vector_nss[:1] -= (a1, a2, c, b) == (1, 1, 1, 1)
        lines.append(' '.join(map(str, vector_nss)) + ' 0')

    # 5) Inequalities
    lines.append('Inequalities:')
    # the ineqs are -p(a1a2cb|x1x2y) <= 0 for all a1,a2,c,b,x1,x2,y. For the final 8 rows, note that this means -(p(1111|x1x2y) - 1) - 1 <= 0.
    # TODO could clean this up by using homogeneous coords in all relevant functions
    NtoF = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2).astype('int8')
    dim_full = 128
    for i in range(0, dim_full):
        lines.append(' '.join(map(str, -NtoF[i])) + (' 0' if i < 120 else ' -1'))

    # Write to file
    if filename is None:
        filename = 'panda-files/arc-output/job31/tmp.pi'
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


def convert_panda_output_vertex_to_input_vertex_format(old_filename, new_filename):
    old_file = open(old_filename, 'r')
    new_lines = []
    for line in old_file.readlines():
        try:
            new_lines.append(row_with_denom_to_vector_str(list(map(int, line.split()))))
        except ValueError:
            # Line was not filled with integers. Probably a comment/header line. Leave intact but remove the newline.
            new_lines.append(line.strip())
            pass
    old_file.close()
    new_file = open(new_filename, 'w')
    new_file.write('\n'.join(new_lines))
    new_file.close()


def row_with_denom_to_vector_str(row):
    from fractions import Fraction
    vector = list(row[:-1])
    denominator = row[-1]
    for i in range(len(vector)):
        vector[i] = str(Fraction(vector[i], denominator))
    return ' '.join(vector)


def row_with_denom_to_vector(row):
    return 1 / row[-1] * np.array(row[:-1])
