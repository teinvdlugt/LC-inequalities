import sys

import scipy.linalg
import sympy
from scipy.optimize import linprog
import time, datetime

import panda
import utils
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
    for line in open('panda-files/results/lc_vertices', 'r').readlines():
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
    """ TODO this function can be massively improved by using that affine indep <=> lin indep of the homogenised coordinates (rather than using LP).
    NOTE but also, this function wasn't really necessary, because if an affine subspace spanned by an affinely independent set S does not intersect the origin, then S is also linearly independent.
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

            # if not polytope_utils.in_affine_hull_lp(points, point):
            if not in_affine_hull:
                print()
                print("Affinely independent point found on line %d of file (counting from 1)" % (empty_line_count + point_count))
                return point

        line = file.readline()

    print()
    print("Reached end of file. No affinely independent point found!")


def find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q, known_facets=None, check_vertices_are_on_face=True, violation_threshold=1e-10, output_file=None, snapshot_file=None, load_from_snapshot=True,
                                               snapshot_frequency=100, update_frequency_j=100,
                                               carriage_return=True):
    """ TODO can I use integer arithmetic? Calculating nullspaces with homogeneous coordinates etc?
    NOTE I think this function only works for full-dimensional polytopes.
    :param face: length-(d+1) vector representing an inequality that is valid for Q.
    :param P: shape-(d-2,d+1) array; each row is the homogeneous representation of a vector, and the convex hull of the d-2 vectors is `face` ∩ LC (which is d-3 dimensional).
    :param Q: list of vertices to search, sorted according to how likely you think it is that they will span a facet together with P (more likely first).
              This is also the list that will be used to determine which hyperplanes constitute valid inequalities (in the sense that all pts of Q lie in one halfspace).
    :param check_vertices_are_on_face: set to False if you already know that all vertices in Q are NOT on the face defined by P.
    :param violation_threshold: when checking whether a hyperplane constitutes a valid inequality, violation of the inequality is only taken seriously when
                          abs(violation) > violation_tol. A higher violation_tol will reduce the chance of missing facets, but increases the chance of returning false facets.
                          (so do check if all facets are indeed valid for LC afterwards).
    :return: array of shape (?,d+1); each row represents a facet (not symmetry-reduced!) adjacent to the face specified by P. (also prints these rows)
    """
    end = '\r' if carriage_return else '\n'

    d = len(Q[0]) - 1
    assert d == P.shape[1] - 1

    facets = np.array(known_facets, dtype='int64') if known_facets else np.empty((0, d + 1), dtype='int64')

    if output_file is not None:
        with open(output_file, 'a') as f:
            f.write("\n\n--- NEW RUN, %s ---\n" % str(datetime.datetime.now()))
            for facet in facets:
                f.write(' '.join(map(str, facet)) + '\n')

    # Load from snapshot file
    if snapshot_file and load_from_snapshot:
        with open(snapshot_file, 'r') as f:
            lines = f.readlines()
            vertex_candidate_indices = list(map(int, lines[1].split()))
            vertices_not_on_face_indices = list(map(int, lines[3].split()))
            i = int(lines[5])
            # scan for appropriate value of _i
            _i = 0
            for _i in range(0, len(vertex_candidate_indices)):
                if vertex_candidate_indices[_i] >= i:
                    break

    else:
        vertex_candidate_indices = list(range(0, len(Q)))
        vertices_not_on_face_indices = list(range(0, len(Q)))
        _i = 0

    indices_to_try_in_quadrant_method = np.r_[  # I have reason to believe that the first couple vertices in vertices_not_on_face_indices are already sufficient. Also take some random ones.
        vertices_not_on_face_indices[0:min(100, len(vertices_not_on_face_indices))], np.random.choice(vertices_not_on_face_indices, min(100, len(vertices_not_on_face_indices)),
                                                                                                      replace=False)]

    P_sympy = sympy.Matrix(P)

    total_sympy_time = 0
    total_quadrant_time = 0
    secant_vertices_caught = 0  # 'secant vertices': vertices q s.t. <P,q1> passes through the interior of the polytope
    duplicate_facets_caught = 0
    times_not_done_quadrant = 0
    start_time = time.time()

    while _i < len(vertex_candidate_indices):
        i = vertex_candidate_indices[_i]
        if check_vertices_are_on_face and np.dot(face, Q[i]) == 0:  # if Q[i] is already on the face
            vertex_candidate_indices.remove(i)
            vertices_not_on_face_indices.remove(i)
            continue

        P_qi = np.r_[P, [Q[i]]]

        ## Do quadrant method
        # But don't do it if Q[i] is on one of the already found facets! Because then quadrant method will never work
        if np.all(facets @ Q[i]):  # if for all m, facets[m] @ Q[i] != 0
            time1 = time.time()
            a1a2 = scipy.linalg.null_space(P_qi)
            a1 = a1a2[:, 0]
            a2 = a1a2[:, 1]
            # TODO maybe perturb/randomise a1,a2. Test if that makes it faster when running on LC.
            # Loop through a selection of vertices.
            # Try to find one vertex for each quadrant.
            found_quadrant_gt_gt = found_quadrant_gt_lt = found_quadrant_lt_gt = found_quadrant_lt_lt = False
            _k = 0  # counting number of iterations in quadrant method
            for k in indices_to_try_in_quadrant_method:
                _k += 1
                q = Q[k]
                violation1 = np.dot(q, a1)
                if violation1 > violation_threshold and not (found_quadrant_gt_gt and found_quadrant_gt_lt):
                    violation2 = np.dot(q, a2)
                    if (not found_quadrant_gt_gt) and violation2 > violation_threshold:  # TODO remove 'not found_...' etc.?
                        found_quadrant_gt_gt = True
                    elif (not found_quadrant_gt_lt) and violation2 < -violation_threshold:
                        found_quadrant_gt_lt = True
                elif violation1 < -violation_threshold and not (found_quadrant_lt_gt and found_quadrant_lt_lt):
                    violation2 = np.dot(q, a2)  # looks like copied code, but it's for efficiency (don't want to unnecessarily compute violation2)
                    if (not found_quadrant_lt_gt) and violation2 > violation_threshold:
                        found_quadrant_lt_gt = True
                    elif (not found_quadrant_lt_lt) and violation2 < -violation_threshold:
                        found_quadrant_lt_lt = True
                if found_quadrant_gt_gt and found_quadrant_gt_lt and found_quadrant_lt_gt and found_quadrant_lt_lt:
                    # Found that Q[i] is a 'secant vertex'!
                    break
            total_quadrant_time += time.time() - time1

            print("%s; i=%d; facets=%d; candidates=%d; sympy=%.1fs; secant=%.1fs; secant vertices caught=%d; duplicate facets caught=%d; avoided quadrant=%d; quadrant iters=%d"
                  % (datetime.timedelta(seconds=int(time.time() - start_time)), i, len(facets), len(vertex_candidate_indices), total_sympy_time, total_quadrant_time, secant_vertices_caught,
                     duplicate_facets_caught, times_not_done_quadrant, _k),
                  end=end)
            sys.stdout.flush()

            if found_quadrant_gt_gt and found_quadrant_gt_lt and found_quadrant_lt_gt and found_quadrant_lt_lt:
                vertex_candidate_indices.remove(i)  # NOTE Don't increment _i because that'd result in skipping every other vertex
                secant_vertices_caught += 1
                continue
        else:
            times_not_done_quadrant += 1
            print("%s; i=%d; facets=%d; candidates=%d; sympy=%.1fs; secant=%.1fs; secant vertices caught=%d; duplicate facets caught=%d; avoided quadrant=%d"
                  % (datetime.timedelta(seconds=int(time.time() - start_time)), i, len(facets), len(vertex_candidate_indices), total_sympy_time, total_quadrant_time, secant_vertices_caught,
                     duplicate_facets_caught, times_not_done_quadrant),
                  end=end)
            sys.stdout.flush()

        # Quadrant method didn't rule out Q[i]; proceed by checking if there's j s.t. P,Q[i],Q[j] span a facet.
        for _j in range(0, len(vertex_candidate_indices)):
            j = vertex_candidate_indices[_j]
            if j >= i:
                break  # only check the 'upper triangle' in (j,i)-space

            P_qi_qj = np.r_[P_qi, [Q[j]]]
            if np.linalg.matrix_rank(P_qi_qj) == d:
                # First check if we don't already have found the facet that contains these two points qi,qj. Doing this should also avoid any duplicates in the facets output
                # This takes O(len(facets)) but potentially saves us O(len(Q)) time
                already_found_this_facet = False
                for facet in facets:
                    if np.dot(facet, Q[i]) == np.dot(facet, Q[j]) == 0:  # TODO maybe abs(np.dot(...)) < violation_threshold? But using sympy this might not be necessary
                        already_found_this_facet = True
                        duplicate_facets_caught += 1
                        break
                if already_found_this_facet:
                    continue

                # Maybe move quadrant method to here, and use qj instead of random ONB of null space

                if _j % update_frequency_j == 0 and j != 0:
                    print("%s; (i,j)=(%d,%d); facets=%d; candidates=%d; sympy=%.1fs; secant=%.1fs; secant vertices caught=%d; duplicate facets caught=%d; avoided quadrant=%d   "
                          % (datetime.timedelta(seconds=int(time.time() - start_time)), i, j, len(facets), len(vertex_candidate_indices), total_sympy_time, total_quadrant_time, secant_vertices_caught,
                             duplicate_facets_caught, times_not_done_quadrant),
                          end=end)
                    sys.stdout.flush()

                # Find normal vector to plane through P,qi,qj:
                a = scipy.linalg.null_space(P_qi_qj)[:, 0]
                # Check if it defines a valid hyperplane
                found_gt0, found_lt0 = None, None
                for k in vertices_not_on_face_indices:
                    violation = np.dot(a, Q[k])
                    if violation > violation_threshold:  # a will likely involve numerical errors - it looks like they're about 1e-16 but let's be careful - o/w we might miss facets of LC
                        found_gt0 = Q[k]
                        if found_lt0 is not None:
                            break  # `a` does not support a valid inequality. Move on to next `m`
                    if violation < -violation_threshold:
                        found_lt0 = Q[k]
                        if found_gt0 is not None:
                            break  # sim.

                if found_gt0 is None or found_lt0 is None:
                    # Success! Found a new facet.
                    # Pretty-print `a` (the inequality) using sympy:   (col_join is like np.r_)
                    time1 = time.time()
                    a = P_sympy.col_join(sympy.Matrix([Q[i], Q[j]])).nullspace()[0]
                    a = np.array(a).T[0]  # get rid of the annoying sympy.Matrix structure
                    total_sympy_time += (time.time() - time1)
                    # using sympy rather than scipy.linalg.null_space here because scipy normalises the vectors, leading to nasty round-off errors
                    # But sympy does cost a lot of time

                    found_vertex = found_gt0 if found_gt0 is not None else found_lt0
                    if np.dot(a, found_vertex) > 0:  # Flip sign of inequality appropriately. Note that sympy `a` might differ from scipy `a`.
                        a = -1 * a

                    facets = np.r_[facets, [a]]
                    a_str = ' '.join(map(str, a))
                    print(a_str)
                    if output_file:
                        with open(output_file, 'a') as f:
                            f.write('+' + a_str + '\n')

        # Save snapshot
        if _i % snapshot_frequency == 0 and output_file:
            with open(output_file + '_snapshot', 'w') as f:
                f.write('vertex_candidate_indices:\n'
                        + ' '.join(map(str, vertex_candidate_indices)) + '\n'
                        + 'vertices_not_on_face_indices:\n'
                        + ' '.join(map(str, vertices_not_on_face_indices)) + '\n'
                        + 'checked i (up to and including):\n%d\n'
                        % i
                        + 'facets:\n')
                for facet in facets:
                    f.write(' '.join(map(str, facet)) + '\n')

        _i += 1

    print("All vertices in Q checked. Found %d facets!" % len(facets))
    print("Summary:\n Elapsed time: %s\n sympy time: %.1fs\n secant time: %.1fs\n vertices caught by quadrant method: %d\n duplicate facets caught: %d\n times not done quadrant method: %d"
          % (datetime.timedelta(seconds=int(time.time() - start_time)), total_sympy_time, total_quadrant_time, secant_vertices_caught, duplicate_facets_caught, times_not_done_quadrant))
    print("WARNING: recall that the returned facets are only confirmed to be valid up to a tolerance of " + str(violation_threshold))
    return facets


def test_find_facets_adjacent_to_d_minus_3_dim_face():
    ## Let's try an octahedron centred around the origin in R^3
    # The vertices (homogeneous coords):
    Q = [[1, 0, 0, 1],
         [0, 1, 0, 1],
         [0, 0, 1, 1],
         [-1, 0, 0, 1],
         [0, -1, 0, 1],
         [0, 0, -1, 1]]
    # The hyperplane that touches the octahedron in one point: e.g. x <= 1  <=>  x - 1 <= 0
    face = [1, 0, 0, -1]
    # point on the face:
    P = np.array([[1, 0, 0, 1]])
    # Run algorithm
    print("Octahedron (should yield 4 facets):")
    facets = list(map(list, find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q, output_file='panda-files/test_find_facets')))
    assert len(facets) == 4
    assert [1, 1, 1, -1] in facets
    assert [1, 1, -1, -1] in facets
    assert [1, -1, 1, -1] in facets
    assert [1, -1, -1, -1] in facets

    ## Let's now try the 0/1 cube
    Q = np.concatenate((list(itertools.product([0, 1], repeat=3)), np.ones((8, 1), 'int8')), axis=1).tolist()
    face = [-1, -1, -1, 0]  # x + y + z >= 0, a plane touching the vertex of Q that is the origin
    P = np.array([[0, 0, 0, 1]])
    print("\nCube (should yield 3 facets):")
    facets = list(map(list, find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q, carriage_return=False)))
    assert len(facets) == 3
    assert [-1, 0, 0, 0] in facets
    assert [0, -1, 0, 0] in facets
    assert [0, 0, -1, 0] in facets

    ## Let's try the opposite vertex of the 0/1 cube
    Q = np.concatenate((list(itertools.product([0, 1], repeat=3)), np.ones((8, 1), 'int8')), axis=1).tolist()
    face = [1, 1, 1, -3]  # x + y + z >= 0, a plane touching the vertex of Q that is the origin
    P = np.array([[1, 1, 1, 1]])
    print("\nCube opposite vertex:")
    facets = list(map(list, find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q)))
    assert len(facets) == 3
    assert [1, 0, 0, -1] in facets
    assert [0, 1, 0, -1] in facets
    assert [0, 0, 1, -1] in facets

    ## Let's see what happens with non-0/1 vertices. Try the cube with side lengths 1/2
    Q = np.concatenate((list(itertools.product([0, 1], repeat=3)), 2 * np.ones((8, 1), 'int8')), axis=1).tolist()
    face = [[2, 2, 2, -3]]  # x + y + z >= 0, a plane touching the vertex of Q that is the origin
    P = np.array([[1, 1, 1, 2]])
    print("\nCube with side-lengths 1/2:")
    facets = list(map(list, find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q)))
    assert len(facets) == 3
    assert [2, 0, 0, -1] in facets
    assert [0, 2, 0, -1] in facets
    assert [0, 0, 2, -1] in facets

    ## Try the d-simplex
    d = 30  # dimension of the simplex. I'm embedding in R^d here, not in R^{d+1} as usual.
    def make_Q():
        Q = np.zeros((d + 1, d + 1), 'int8')
        Q[0][-1] = 1  # first vertex: the origin
        for i in range(1, d + 1):
            Q[i][i - 1] = Q[i][-1] = 1  # remaining vertices: 'one-hot' vectors (i.e. standard basis vectors)
        return Q.tolist()
    # We actually know the facets in this case:
    facets = []
    facets.append(np.r_[[1] * d, [-1]].tolist())  # sum_i x_i <= 1  the 'diagonal facet'
    for i in range(0, d):
        facets.append(np.r_[[0] * i, [-1], [0] * (d - i)].tolist())  # x_i >= 0
    # Any (d-3)-dimensional face is the intersection of 3 facets. Let's be interested specifically in the case where one of these facets is the 'diagonal facet'
    # The other two facets will be x_i >= 0, x_j >= 0 for some i,j. Let's test if the algorithm works for all pairs (i,j)
    for i in range(0, d):
        for j in range(i + 1, d):
            face = np.ones(d + 1, 'int8')
            face[-1] = -1
            face[i] = face[j] = 0  # face: sum_{k!=i,j} x_k <= 1
            Q = make_Q()
            P = [vertex for vertex in Q]
            P.pop(j + 1)
            P.pop(i + 1)
            P.pop(0)
            P = np.array(P)
            print("\n%d-simplex, face with (i,j)=(%d,%d):" % (d, i, j))
            found_facets = list(map(list, find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q)))

            assert len(found_facets) == 3
            assert facets[0] in found_facets  # the 'diagonal facet'
            assert facets[i + 1] in found_facets
            assert facets[j + 1] in found_facets
            print("Success!")  # ✓ (lijkt te werken voor alle d)
            return
    # Let's now check the case where the three facets are all 'x_d >= 0' facets.
    i, j, k = 0, 1, 2
    face = np.zeros(d + 1, 'int8')
    face[-1] = 0
    face[i] = face[j] = face[k] = -1
    Q = make_Q()
    P = [vertex for vertex in Q]
    P.pop(k + 1)
    P.pop(j + 1)
    P.pop(i + 1)
    P = np.array(P)
    print("\n%d-simplex, face with (i,j,k)=(%d,%d,%d):" % (d, i, j, k))
    found_facets = list(map(list, find_facets_adjacent_to_d_minus_3_dim_face(face, P, Q)))
    assert len(found_facets) == 3
    for l in [i, j, k]:
        assert facets[l + 1] in found_facets
    print("Success!")


def is_ineq_valid_for_LC(ineq):
    with open('panda-files/results/8 all LC vertices', 'r') as LC_vertices:
        line = LC_vertices.readline()
        vertex_count = 0
        while line:
            if line.strip():
                vertex = list(map(int, line.split()))
                vertex_count += 1
                if np.dot(ineq, vertex) > 0:
                    print("No, the following vertex violates the provided inequality by %f:" % np.dot(ineq, vertex))
                    print(' '.join(map(str, vertex)))
                    return False
    print("The inequality is valid for LC!")
    return True


def is_facet_of_LC(ineq):
    """ Checks whether ineq is valid for LC and whether LC ∩ ineq is 85-dimensional. """
    max_dimension = 85

    with open('panda-files/results/8 all LC vertices', 'r') as LC_vertices:
        aff_indep_subset = np.empty((0, 87), 'int8')

        vertex_count = 0

        line = LC_vertices.readline()
        while line:
            if line.strip():
                vertex = list(map(int, line.split()))
                violation = np.dot(vertex, ineq)
                if violation > 0:
                    print("The inequality")
                    print(' '.join(map(str, ineq)))
                    print("is NOT valid for LC, as it is violated by the LC vertex")
                    print(' '.join(map(str, vertex)))
                    return False
                if len(aff_indep_subset) - 1 < max_dimension and violation == 0:
                    # vertex is on the hyperplane defined by the inequality
                    # Check if vertex is not already in span of previously found ones
                    new_matrix = np.r_[aff_indep_subset, [vertex]]
                    if np.linalg.matrix_rank(new_matrix) == len(aff_indep_subset) + 1:  # i.e. if matrix has full rank
                        aff_indep_subset = new_matrix
                vertex_count += 1

            print("Processed %d vertices; currently at dimension %d" % (vertex_count, len(aff_indep_subset) - 1), end='\r')

            line = LC_vertices.readline()

    print()
    if len(aff_indep_subset) == 0:
        print("The inequality is valid for LC but is not a strict inequality and hence does not support a face of LC.")
        return False
    elif len(aff_indep_subset) - 1 == max_dimension:
        print("The inequality supports a facet of LC!")
        return True
    elif len(aff_indep_subset) - 1 > max_dimension:
        print("Something weird just happened.")
        return False
    else:
        print("The inequality supports a face of LC of dimension %d" % (len(aff_indep_subset) - 1))
        return False


def count_LC_vertices_on_facets(facets):
    """ not entirely finished. commented bit is more efficient than current code """
    vertex_counts = [0] * len(facets)
    vertices_not_on_a_facet = []
    facets = np.array(facets)

    with open('panda-files/results/8 all LC vertices') as f:
        batch_size = 10000
        line = f.readline()
        while line:
            """# load batch
            current_batch = []
            while line and len(current_batch) < batch_size:
                if line.strip():
                    current_batch.append(list(map(int, line.split())))
                line = f.readline()
            current_batch = np.array(current_batch).T

            # process batch
            violations = facets @ current_batch   # will be  len(facets) x batch_size  matrix
            for i in range(0, len(facets)):
                vertex_counts[i] += np.sum(violations[i] == 0)   # the i-th row of violations has all the violations of facets[i]
            # the vertices that are on none of the facets correspond to the columns of violations with all non-zero entries
            for i in np.where(np.all(violations, axis=0)):   # np.where(np.all([[1,2,3,0],[1,2,3,4]], axis=0)) gives [0,1,2]
                vertices_not_on_a_facet.append(current_batch[:,i])"""

            if line.strip():
                vertex = list(map(int, line.split()))
                on_a_facet = False
                for i in range(0, len(facets)):
                    if np.dot(facets[i], vertex) == 0:
                        vertex_counts[i] += 1
                        on_a_facet = True
                if not on_a_facet:
                    vertices_not_on_a_facet.append(vertex)
            print('not on a facet: %d, on each facet: %s' % (len(vertices_not_on_a_facet), str(vertex_counts)), end='\r')
            sys.stdout.flush()
            line = f.readline()
            if len(vertices_not_on_a_facet) > 100:
                break
    return vertex_counts, vertices_not_on_a_facet


def violations_of_known_lc_facets(cor_nss, known_lc_facets_file='panda-files/results/14 lc_known_facets'):
    """
    :param cor_nss: length-87 vector of homogeneous NSS coordinates.
    :param known_lc_facets_file: filename of file to load known LC facets from.
    :return: list of violations of all known LC facets by cor_nss.
    """
    known_lc_facets = utils.read_vertex_range_from_file(known_lc_facets_file)
    assert len(known_lc_facets == 8)

    return (1 / cor_nss[-1]) * known_lc_facets @ cor_nss


def maximum_violation_of_known_lc_facets(cors, known_lc_facets_file='panda-files/results/14 lc_known_facets'):
    """
    :param cors: each element should have length 87: NSS representation, homogeneous vector
    """
    known_lc_facets = utils.read_vertex_range_from_file(known_lc_facets_file)
    cors = cors.astype('float')
    for i in range(0, len(cors)):  # normalise the qm cors
        cors[i] = (1. / cors[i][-1]) * cors[i]
    return np.max(known_lc_facets @ cors.T)


def maximum_violation_of_caus2_facets(cors):
    caus2_facets = utils.read_vertex_range_from_file('panda-files/results/15 Caus2/all_caus2_facets')
    cors = cors.astype('float')
    for i in range(0, len(cors)):  # normalise the qm cors
        cors[i] = (1. / cors[i][-1]) * cors[i]
    return np.max(caus2_facets @ cors.T)


def a_p_that_is_not_in_LC():
    """ Returns the correlation on [p200], in homogeneous NSS coordinates. """
    p1 = np.zeros((2,) * 7, 'int')
    p2 = np.zeros((2,) * 7, 'int')
    for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
        if c == 0 and a2 == x1 and a1 == 0 and b == x1:
            p1[a1, a2, c, b, x1, x2, y] = 1
        if c == 0 and a2 == 0 and a1 == x2 and b == 1 - x1:
            p2[a1, a2, c, b, x1, x2, y] = 1

    p = 1 / 2 * (p1 + p2)
    p_homog = np.r_[p.reshape(128), [1]]
    assert vs.is_in_NSS(p_homog, 8, 2, 4, 2)
    return vs.construct_full_to_NSS_homog(8, 2, 4, 2) @ p_homog


def deterministic_cor_full_homog(a1, a2, c, b):
    """ a1, a2, c, b should be functions (e.g. lambdas) from three binary variables to one binary variable. The returned vector is full homogeneous, so of length 129. """
    cor = np.zeros((2,) * 7, dtype='int')
    for _a1, _a2, _c, _b, x1, x2, y in itertools.product((0, 1), repeat=7):
        cor[_a1, _a2, _c, _b, x1, x2, y] = 1 * (_a1 == a1(x1, x2, y)) * (_a2 == a2(x1, x2, y)) * (_c == c(x1, x2, y)) * (_b == b(x1, x2, y))
    return np.r_[cor.reshape((128,)), [1]]


if __name__ == '__main__':
    # qm_cor_str = qm.quantum_cor_in_panda_format_nss(
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

    # Trying to find LC vertex on GYNI plane that is affinely independent to those in result file 10 (used in job18). Later realised there was no hope in finding any.
    """
    gyni = inequality_GYNI()
    result = find_affinely_independent_point(
        points=np.array([list(map(int, line.split())) for line in open('panda-files/results/10 lin indep on GYNI').readlines()[5:]]),
        file_to_search='panda-files/results/8 all LC vertices',
        constraint=lambda row: np.dot(gyni, row) == 0
    )
    print(result)
    """

    # test_find_facets_adjacent_to_d_minus_3_dim_face()
    # assert 2 + 2 == 5

    """
    # P = result file 10 but homogenised
    P = np.concatenate((utils.read_vertex_range_from_file('panda-files/results/10 lin indep on GYNI'), np.ones((84, 1), 'int16')), axis=1)
    # Q = vertices to search = result file 8
    # Q = np.r_[
    #     utils.read_vertex_range_from_file('panda-files/results/8 all LC vertices', 50176, 54272),
    #     utils.read_vertex_range_from_file('panda-files/results/8 all LC vertices', 0, 1024)
    # ]
    # or
    Q = np.load('panda-files/results/lc_vertices.npy')
    # Q = np.flipud(Q)  # if you want to go through Q in reverse order
    # np.random.shuffle(Q)  # if you want to go through Q in random order
    print("Loaded P and Q into memory")
    facets = find_facets_adjacent_to_d_minus_3_dim_face(inequality_GYNI(), P, Q, known_gyni_facets,
                                                        # output_file='panda-files/results/12 facets adjacent to GYNI',
                                                        # snapshot_file='panda-files/results/12 facets adjacent to GYNI_snapshot',
                                                        carriage_return=False)
    """

    ## To get all LC vertices NOT on GYNI (but maybe they _are_ on a face that is mapped to GYNI under a symmetry of LC!):
    """
    gyni = inequality_GYNI()
    vector_space_utils.filter_row_file('panda-files/results/8 all LC vertices', 'panda-files/lc_vertices_not_satisfying_random_ineq', lambda row: np.dot(random_ineq, row) > 0)
    """

    """
    for facet in [*known_gyni_facets, *known_lgyni_facets]:
        qm_violations = qm.some_quantum_violations(facet)
        print("Maximum violation found: %f" % np.max(qm_violations))
        if np.any(qm_violations > 0):
            print("VIOLATION!!!")
    """

    # Try the following distr which I thought of in notes [around p200]
    """violations = violations_of_known_lc_facets(a_p_that_is_not_in_LC())"""

    # maybe_facets = [list(map(int,
    #                          "-1 0 0 1 -1 0 0 1 0 0 -1 1 0 0 -1 1 0 -1 0 1 0 -1 0 1 0 0 0 0 -2 -2 2 2 0 0 0 0 -2 -2 2 2 0 0 0 0 -2 -2 0 0 0 0 2 2 -2 -2 0 0 0 0 2 2 -2 -2 0 0 2 2 0 0 -2 -2 0 0 2 2 0 0 -2 -2 0 0 0 0 0 0 0 0 1".split())),
    #                 list(map(int,
    #                          "1 0 0 -1 1 0 0 -1 0 0 1 -1 0 0 1 -1 0 1 0 -1 0 1 0 -1 0 0 0 0 -2 2 2 -2 0 0 0 0 -2 2 2 -2 0 0 0 0 -2 2 0 0 0 0 2 -2 -2 2 0 0 0 0 2 -2 -2 2 0 0 2 -2 0 0 -2 2 0 0 2 -2 0 0 -2 2 0 0 0 0 0 0 0 0 -1".split())),
    #                 list(map(int,
    #                          "1 0 0 -1 1 0 0 -1 0 0 1 -1 0 0 1 -1 0 1 0 -1 0 1 0 -1 0 0 0 0 2 -2 -2 2 0 0 0 0 2 -2 -2 2 0 0 0 0 2 -2 0 0 0 0 -2 2 2 -2 0 0 0 0 -2 2 2 -2 0 0 -2 2 0 0 2 -2 0 0 -2 2 0 0 2 -2 0 0 0 0 0 0 0 0 -1".split())),
    #                 # list(map(int,
    #                 #          "1 0 0 -1 1 0 0 -1 0 0 1 -1 0 0 1 -1 0 1 0 -1 0 1 0 -1 0 0 0 0 2/3 2/3 -2/3 -2/3 0 0 0 0 2/3 2/3 -2/3 -2/3 0 0 0 0 2/3 2/3 0 0 0 0 -2/3 -2/3 2/3 2/3 0 0 0 0 -2/3 -2/3 2/3 2/3 0 0 -2/3 -2/3 0 0 2/3 2/3 0 0 -2/3 -2/3 0 0 2/3 2/3 0 0 0 0 0 0 0 0 -1".split()))
    #                 ]
    #
    #
    # for facet in maybe_facets:
    #     print("checking facet %s" % ' '.join(map(str, facet)))
    #     is_facet_of_LC(facet)

    # Checking computation of [p218]. this is lc_abxy.facet1()
    ineq = list(map(int,
                    '1 0 0 -1 1 0 0 -1 0 0 1 -1 0 0 1 -1 0 1 0 -1 0 1 0 -1 0 0 0 0 0 1 0 -1 0 0 0 0 0 1 0 -1 0 0 0 0 0 1 0 0 0 0 0 -1 0 1 0 0 0 0 0 -1 0 1 0 0 0 -1 0 0 0 1 0 0 0 -1 0 0 0 1 0 0 0 0 0 0 0 0 -1'.split()))
    caus2_cor_full = 1 / 2 * deterministic_cor_full_homog(lambda x1, x2, y: 0,  # a1
                                                          lambda x1, x2, y: x1,  # a2
                                                          lambda x1, x2, y: 0,  # c
                                                          lambda x1, x2, y: (x2 + 1) % 2) + \
                     1 / 2 * deterministic_cor_full_homog(lambda x1, x2, y: x2,  # a1
                                                          lambda x1, x2, y: 0,  # a2
                                                          lambda x1, x2, y: 0,  # c
                                                          lambda x1, x2, y: x2)  # b
    assert vs.is_in_NSS(caus2_cor_full, 8, 2, 4, 2)
    caus2_cor_nss = vs.construct_full_to_NSS_homog(8, 2, 4, 2) @ caus2_cor_full
    print('Violation:', caus2_cor_nss @ ineq)
