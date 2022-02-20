import sys

import panda
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

    ## WRITE MAPS
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


if __name__ == '__main__':
    write_panda_file_just_for_maps('panda-files/h_symms.pi', symmetry_utils.H_symm_generators(), 86)
    write_panda_file_just_for_maps('panda-files/lc_symms.pi', symmetry_utils.LC_symm_generators(), 86)
