import sys

from scipy.optimize import linprog

import panda
import polytope_utils
import quantum_utils
import utils
from quantum_utils import proj, kron, ket_plus, phi_plus, z_onb, x_onb, ket0
import symmetry_utils
import itertools
import numpy as np
import vector_space_utils as vs

import vector_space_utils


def LC_NSCO1_symms_left_coset_reps():
    """ See [p159-162] """
    alpha_tuples = list(itertools.product((0, 1), repeat=10))
    # Remove redundant ones. See [p162] (on the page where Conj-18-2-22 is)
    i = 0
    while i < len(alpha_tuples):
        alpha_tuple = alpha_tuples[i]
        (al, be, ga, de, ep, ze, et, th, ka, la) = alpha_tuple
        equivalent_alpha_tuple = tuple(map(lambda x: x % 2,
                                           (al, be, ga, de + al, ep + be, ze + 1, et + ga, th, ka, la)))
        if equivalent_alpha_tuple in alpha_tuples:
            assert alpha_tuples.index(equivalent_alpha_tuple) > i  # if equivalent_alpha_tuple has already been checked in an earlier step then it should have removed alpha_tuple in that step
            # Remove equivalent_alpha_tuple from alpha_tuples
            alpha_tuples.remove(equivalent_alpha_tuple)
        else:
            print('No counterpart found')
        i += 1
    # print('|H\H^c| =', len(alpha_tuples))  # gives 512 as expected

    # Each alpha_tuple corresponds to a representative of a class in the left quotient H\H^c. Now construct those representatives from alpha_tuples
    # explicitly. [p162]
    result = []
    for alphas in alpha_tuples:
        result.append(symmetry_utils.nss_var_perm_to_symm(
            lambda a1, a2, c, b, x1, x2, y: (a1,
                                             (a2 + alphas[3] * x1 + alphas[4] * a1 + alphas[5] * x2 + alphas[6] * x1 * a1 + alphas[7] * x1 * x2 + alphas[8] * a1 * x2 + alphas[9] * x1 * a1 * x2) % 2,
                                             c, b, x1,
                                             (x2 + alphas[0] * x1 + alphas[1] * a1 + alphas[2] * x1 * a1) % 2,
                                             y)))

    # Test whether there are duplicate symmetries
    # assert np.unique(np.array(result), axis=0).shape[0] == len(result)

    return result


def write_unpacking_input_file(map_filename='panda-files/unpacking_maps_1024', vertex_filename='panda-files/unpacking_vertices'):
    """Write the input file necessary for the customised PANDA code which lists all lc-classes that the given nsco1-classes of nsco1 vertices fall apart in."""
    var_names = ['x' + str(i) for i in range(0, 86)]

    ## WRITE MAPS
    map_file = open(map_filename, 'w')
    map_file.write('DIM=86\nNames:\n')
    map_file.write(' '.join(var_names) + '\nMaps:\n')

    # The first 1024 maps will be representatives of the left coset classes in H\H^c
    left_coset_reps = LC_NSCO1_symms_left_coset_reps()
    for symm in left_coset_reps:
        map_file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')
    # The next 7 maps will be the lc symmetries
    lc_symms = symmetry_utils.H_symm_generators()
    for symm in lc_symms:
        map_file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')
    # The next 7 maps will be the nsco1 symmetries - but expressed in NSS coordinates!
    nsco1_symms = symmetry_utils.nsco1_symm_generators_in_nss_coords()
    for symm in nsco1_symms:
        map_file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')

    # For the data, just some boilerplate that I won't use anyway:
    map_file.write(
        "Inequalities:\n1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
    map_file.close()

    ## WRITE_VERTICES
    vertex_file = open(vertex_filename, 'w')
    vertex_file.write("CONV_SECTION\n")

    # The 56 vertex classes of NSCO1, expressed in NSS coords.
    # TODO later: But to allow us to get a nice informative output in C++, let's also add all of those in their original NSCO1 coords.
    rows_in_nsco1_coords = []
    rows_in_nss_coords = []
    nsco1_file = open('panda-files/nsco1_vertex_classes.out', 'r')
    for line in nsco1_file.readlines():
        row_in_nsco1_coords = list(map(int, line.split()))
        row_in_nss_coords = vector_space_utils.NSCO1_to_NSS_with_denominator(row_in_nsco1_coords)
        # Convert to Fraction format. TODO is that necessary? Cleaner if not do this
        rows_in_nsco1_coords.append(panda.row_with_denom_to_vector_str(np.r_[row_in_nsco1_coords, [4, ] * 6]) + '\n')  # Note that we're appending '4 4 4 4 4 4' because we have to fill up to dim 86
        rows_in_nss_coords.append(panda.row_with_denom_to_vector_str(row_in_nss_coords) + '\n')

        # To get with denominators instead of Fraction format, use:
        # rows_in_nsco1_coords.append(' '.join(map(str, np.r_[row_in_nsco1_coords, [4, ] * 6])) + '\n')
        # rows_in_nss_coords.append(' '.join(map(str, row_in_nss_coords)) + '\n')
    nsco1_file.close()

    for line in rows_in_nsco1_coords:
        vertex_file.write(line)
    for line in rows_in_nss_coords:
        vertex_file.write(line)

    vertex_file.write("END")
    vertex_file.close()
    print('Done writing files')


def do_unpacking_in_python():
    lc_symms = symmetry_utils.H_symm_generators()
    # nsco1_symms = symmetry_utils.nsco1_symm_generators_in_nss_coords()
    left_coset_reps = LC_NSCO1_symms_left_coset_reps()

    nsco1_vertex_classes_nsco1_coords = []
    nsco1_file = open('nsco1_vertex_classes.out', 'r')
    for line in nsco1_file.readlines():
        nsco1_vertex_classes_nsco1_coords.append(list(map(int, line.split())))
    nsco1_file.close()

    def write_line(str):
        output_file = open('unpacked.out', 'a+')
        output_file.write(str + '\n')
        output_file.close()
    def write_line2(str):
        all_lc_classes_file = open('unpacked_all_lc_classes.out', 'a+')
        all_lc_classes_file.write(str + '\n')
        all_lc_classes_file.close()

    v = 0
    for nsco1_vertex_rep_nsco1_coords in nsco1_vertex_classes_nsco1_coords:
        nsco1_vertex_rep_nss_coords = vector_space_utils.NSCO1_to_NSS_with_denominator(nsco1_vertex_rep_nsco1_coords)

        v += 1
        write_line('\nNSCO1 vertex class #' + str(v) + '\n' + ' '.join(map(str, nsco1_vertex_rep_nsco1_coords)))
        write_line('which in NSS coords is\n' + ' '.join(map(str, nsco1_vertex_rep_nss_coords)))

        lc_classes = []
        lc_class_sizes = []  # i-th element is the size of the class represented by lc_classes[i]
        i = 0
        for left_coset_rep in left_coset_reps:
            # sys.stdout.write('\rChecking left-coset representative #%i' % i)
            class_size, lc_rep = symmetry_utils.get_class_representative(left_coset_rep @ nsco1_vertex_rep_nss_coords, lc_symms, False)
            lc_rep = list(lc_rep)
            if lc_rep not in lc_classes:
                lc_classes.append(lc_rep)
                lc_class_sizes.append(class_size)
                write_line2(' '.join(map(str, lc_rep)))
            # else:
            #     utils.assert_soft(lc_class_sizes[lc_classes.index(lc_rep)] == class_size)
            i += 1
        assert len(lc_classes) == len(lc_class_sizes)
        # sys.stdout.write('\r ')
        # sys.stdout.flush()
        write_line('falls apart in the following ' + str(len(lc_classes)) + ' lc-inequivalent classes:')
        for i in range(len(lc_classes)):
            write_line(' '.join(map(str, lc_classes[i])) + ', class size ' + str(lc_class_sizes[i]))

        write_line2('\n')


def write_panda_input_for_nsco1_but_NSS_coords_and_H_symms(filename='panda-files/nsco1_facets_NSS_coords_H_symms.pi'):
    """ Same as panda.nsco1_write_panda_input, but then in NSS coordinates and with just the H symmetries (i.e. those where
     a2,x2 are not controlled by a1,x1, as opposed to panda.nsco1_write_panda_input."""
    lines = []

    # 1) Dimension information
    dim_NSS = 86
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in symmetry_utils.H_symm_generators():
        lines.append(symmetry_utils.symm_matrix_to_string(symm, var_names))

    # 5) Inequalities and equations
    lines.append('Inequalities:')
    # the ineqs are -p(a1a2cb|x1x2y) <= 0 for all a1,a2,c,b,x1,x2,y. For the final 8 rows, note that this means -(p(1111|x1x2y) - 1) - 1 <= 0.
    NtoF = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2).astype('int8')  # list of 'full vectors' expressed in terms of NSS vectors
    for i in range(0, 128):
        lines.append(' '.join(map(str, -NtoF[i])) + (' 0' if i < 120 else ' -1'))

    # Write to file
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


def write_panda_file_just_for_maps(filename, symms, dim):
    var_names = ['x' + str(i) for i in range(0, dim)]

    map_file = open(filename, 'w')
    map_file.write('DIM=%i\nNames:\n' % dim)
    map_file.write(' '.join(var_names) + '\nMaps:\n')

    for symm in symms:
        map_file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')

    # For the data, just some boilerplate that I won't use anyway:
    map_file.write(
        "Inequalities:\n1 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0")
    map_file.close()

    print('Done writing file')


def lc_write_panda_input(filename='panda-files/lc_vertices.pi', shuffle_vertices=False):
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    dim_NSS = 86
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in symmetry_utils.LC_symm_generators():
        lines.append(symmetry_utils.symm_matrix_to_string(symm, var_names))

    # 5) Now we're doing facet enumeration!
    lines.append('Reduced Vertices:')
    vertices = []
    for line in open('panda-files/lc_vertices', 'r').readlines():
        vertices.append(panda.row_with_denom_to_vector_str(list(map(int, line.split()))))
    if shuffle_vertices:
        import random
        random.shuffle(vertices)
    for vertex in vertices:
        lines.append(vertex)

    # Write to file
    if filename is None:
        filename = 'panda-files/old-and-irrelevant/nsco1_facets_perm6feb.pi'
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


def inequality_GYNI():
    """ Bran+15 Eq. (4), but both sides multiplied by 4.
    The inequality is returned in the form of a length-87 vector, where the last element is the negated value of the upper bound (i.e. how PANDA requires its input). """
    coeffs = np.zeros(86)
    upper_bound = 2
    # Literally code Bran+15 Eq. (4)
    NSS_to_full_weird = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2)
    for a1, a2, x1, x2 in itertools.product((0, 1), repeat=4):
        if a1 == x2 and a2 == x1:
            # Need to add p(a1 a2 | x1 x2) in NSS coords to coeffs.
            for _c, _b, _y in itertools.product((0, 1), (0, 1), (0,)):
                coeffs += NSS_to_full_weird[vs.concatenate_bits(a1, a2, _c, _b, x1, x2, _y)]
                upper_bound -= (a1, a2, _c, _b) == (1, 1, 1, 1)  # to correct for weirdness
    result = np.r_[coeffs, [-upper_bound]].astype('int8')

    # Try manually
    manually = np.zeros(87, dtype='int8')
    for i in [0, 4, 17, 21, 10, 14, 27]:
        manually[i] += 1
    for i in [3, 7, 11, 15, 19, 23, 27]:
        manually[i] -= 1
    manually[-1] = -1
    assert np.all(result == manually)

    return result


def inequality_LGYNI():
    """ Bran+15 Eq. (5), but both sides multiplied by 4 """
    coeffs = np.zeros(86)
    upper_bound = 3
    # Literally code Bran+15 Eq. (4)
    NSS_to_full_weird = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2)
    for a1, a2, x1, x2 in itertools.product((0, 1), repeat=4):
        if x1 * (a1 + x2) % 2 == 0 and x2 * (a2 + x1) % 2 == 0:
            # Need to add p(a1 a2 | x1 x2) in NSS coords to coeffs.
            for _c, _b, _y in itertools.product((0, 1), (0, 1), (0,)):
                coeffs += NSS_to_full_weird[vs.concatenate_bits(a1, a2, _c, _b, x1, x2, _y)]
                upper_bound -= (a1, a2, _c, _b) == (1, 1, 1, 1)  # to correct for weirdness
    result = np.r_[coeffs, [-upper_bound]].astype('int8')

    return result


def inequality_violation(vector, inequality):
    """ vector: shape (86,), inequality: shape (87,)
    :returns sth <0 if the vector satisfies the inequality with equality;
             0 if the vector satisfies the inequality strictly;
             sth >0 if the vector does not satisfy the inequality. """
    return np.dot(inequality, np.r_[vector, [1]])


def find_affinely_independent_point(points, file_to_search, constraint=None, update_interval=5000):
    """
    :param points: an array of d-dimensional vectors.
    :param file_to_search: File where each non-empty line is a vector-with-denominator, i.e. d+1 integers.
    :param constraint: a function taking length-d+1 int arrays to booleans. Any line in file_to_search that does
                       not satisfy this constraint will be ignored (if constraint is not None).
    :param update_interval: Print a progress message every update_interval points. None if don't want to print update messages.
    :return: if it exists, a d-dimensional vector that was in `file_to_search`, that satisfies constraints and that is
             not in the affine hull of `points`.
    """
    d = len(points[0])
    file = open(file_to_search, 'r')

    empty_line_count = 0
    point_count = 0

    # For the LP program:
    n = len(points)
    A = np.r_[points.T, np.ones((1, n))]

    line = file.readline()
    while line:
        if not line.strip():
            empty_line_count += 1
            if update_interval is not None:
                print("Empty lines encountered: %d, points processed: %d" % (empty_line_count, point_count), end='\r')
            sys.stdout.flush()
            line = file.readline()
            continue

        row = list(map(int, line.split()))
        point_count += 1

        if constraint is None or constraint(row):
            point = 1. / row[-1] * np.array(row[:-1])
            assert len(point) == d

            if update_interval is not None and point_count % update_interval == 0:
                print("Empty lines encountered: %d, points processed: %d" % (empty_line_count, point_count), end='\r')
            sys.stdout.flush()

            b = np.r_[point, np.ones(1)]
            in_affine_hull = linprog(np.zeros(n), A_eq=A, b_eq=b, options={'tol': 1e-8}, bounds=(None, None)).success  # NOTE using the default tolerance value here. Might want to decrease

            # if not polytope_utils.in_affine_hull(points, point):
            if not in_affine_hull:
                print()
                print("Affinely independent point found on line %d of file (counting from 1)" % (empty_line_count + point_count))
                return point

        line = file.readline()

    print()
    print("Reached end of file. No affinely independent point found!")


if __name__ == '__main__':
    # qm_cor_str = quantum_utils.quantum_cor_in_panda_format_nss(
    #     rho_ctb = proj(kron(ket0, phi_plus).reshape(2,2,2).swapaxes(1,2).reshape(8)), # rho_ctb=proj(kron(ket_plus, phi_plus)),
    #     X1=[z_onb, x_onb],
    #     X2=[z_onb, x_onb],
    #     Y=[z_onb, x_onb],
    #     c_onb=x_onb)
    # print(qm_cor_str)
    # for _ in range(10):
    #     var_values = np.random.randint(0, 2, 7)
    #     print("p(%d,%d,%d,%d|%d,%d,%d) = %s" % (*var_values, qm_cor_str.split()[vector_space_utils.concatenate_bits(*var_values)]))

    # ineq = inequality_GYNI()
    # vectors = np.array([panda.row_with_denom_to_vector(list(map(int, line.split())))
    #            for line in open('panda-files/lc_vertices', 'r').readlines()])
    # violations = []
    # for vector in vectors:
    #     violations.append(inequality_violation(vector, ineq))
    # print(violations)
    # violations = np.array(violations)
    # if len(np.argwhere(violations > 0)) != 0:
    #     print('VIOLATION! Of the following inequalities:', np.argwhere(violations > 0))
    # # --> 33 vertex class reps that satisfy GYNI with equality

    # For result files 10 & 11: (for 11, change GYNI to LGYNI)
    """
    gyni = inequality_GYNI()
    vector_space_utils.reduce_file_to_lin_indep_subset("panda-files/results/8 all LC vertices", 2, 86,
                                                       constraint=lambda row: np.dot(gyni, row) == 0,
                                                       output_filename='panda-files/results/11 lin indep on GYNI, not LGYNI')
    """

    gyni = inequality_GYNI()
    result = find_affinely_independent_point(
        points=np.array([list(map(int, line.split())) for line in open('panda-files/results/10 lin indep on GYNI').readlines()[5:]]),
        file_to_search='panda-files/results/8 all LC vertices',
        constraint=lambda row: np.dot(gyni, row) == 0
    )
    print(result)
