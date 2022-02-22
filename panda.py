import itertools
import numpy as np

import vector_space_utils as vs

B = (0, 1)

import symmetry_utils


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
        vector[i] = str(Fraction(vector[i] / denominator))
    return ' '.join(vector)

def row_with_denom_to_vector(row):
    return 1/row[-1] * np.array(row[:-1])

if __name__ == '__main__':
    pass