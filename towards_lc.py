import sys

import panda
import symmetry_utils
import itertools

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
        if alpha_tuple != equivalent_alpha_tuple and equivalent_alpha_tuple in alpha_tuples:
            assert alpha_tuples.index(equivalent_alpha_tuple) > i  # if equivalent_alpha_tuple has already been checked in an earlier step then it should have removed alpha_tuple in that step
            # Remove equivalent_alpha_tuple from alpha_tuples
            alpha_tuples.remove(equivalent_alpha_tuple)
        else:
            print('No counterpart found')
        i += 1
    print('|H\H^c| =', len(alpha_tuples))

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
    return result


def write_unpacking_input_file(filename='panda-files/unpacking'):
    """Write the input file necessary for the customised PANDA code which lists all lc-classes that the given nsco1-classes of nsco1 vertices fall apart in."""
    file = open(filename, 'w')
    var_names = ['x' + str(i) for i in range(0, 86)]

    file.write('DIM=86\nNames:\n')
    file.write(' '.join(var_names) + '\nMaps:\n')

    # The first 512 maps will be representatives of the left coset classes in H\H^c
    left_coset_reps = LC_NSCO1_symms_left_coset_reps()
    for symm in left_coset_reps:
        file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')
    # The next 7 maps will be the lc symmetries
    lc_symms = symmetry_utils.lc_symm_generators()
    for symm in lc_symms:
        file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')
    # The next 7 maps will be the nsco1 symmetries - but expressed in NSS coordinates!
    nsco1_symms = symmetry_utils.nsco1_symm_generators_in_nss_coords()
    for symm in nsco1_symms:
        file.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')

    # The actual input data will be the 56 vertex classes of NSCO1, expressed in NSS coords.
    # TODO later: But to allow us to get a nice informative output in C++, let's also add all of those in their original NSCO1 coords.
    file.write('Vertices:\n')
    nsco1_file = open('panda-files/nsco1_vertex_classes.out', 'r')
    for line in nsco1_file.readlines():
        row = list(map(int, line.split()))
        row_in_NSS_coords = vector_space_utils.NSCO1_to_NSS_with_denominator(row)
        # Convert to Fraction format. TODO is that necessary? Cleaner if not do this
        file.write(panda.row_with_denom_to_vector_str(row_in_NSS_coords) + '\n')

        # To get with denominators instead of Fraction format, use:
        # file.write('  ' + '  '.join(map(str, row_in_NSS_coords)) + '\n')
    nsco1_file.close()

    file.close()
    print('Done writing file')


if __name__ == '__main__':
    lc_symms = symmetry_utils.lc_symm_generators()
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


    for nsco1_vertex_rep_nsco1_coords in nsco1_vertex_classes_nsco1_coords:
        nsco1_vertex_rep_nss_coords = vector_space_utils.NSCO1_to_NSS_with_denominator(nsco1_vertex_rep_nsco1_coords)

        write_line('The NSCO1 vertex class\n' + ' '.join(map(str, nsco1_vertex_rep_nsco1_coords)))
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
            # else:
            #     utils.assert_soft(lc_class_sizes[lc_classes.index(lc_rep)] == class_size)
            i += 1
        assert len(lc_classes) == len(lc_class_sizes)
        # sys.stdout.write('\r ')
        # sys.stdout.flush()
        write_line('falls apart in the following ' + str(len(lc_classes)) + ' lc-inequivalent classes:')
        for i in range(len(lc_classes)):
            write_line(' '.join(map(str, lc_classes[i])) + ', class size ' + str(lc_class_sizes[i]))

