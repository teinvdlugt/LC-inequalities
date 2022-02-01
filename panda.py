import itertools
import numpy as np
import polytope_utils


def write_panda_input_for_nss(na, nb, nx, ny, filename=None):
    """ See http://comopt.ifi.uni-heidelberg.de/software/PANDA/format.html for format of PANDA input file.
    :param filename: if None, then automatically generated. """
    lines = []  # Store lines here, then write all lines at once at the end of this function.

    # 1) Dimension information
    dim_NSS = polytope_utils.dim_NSS(na, nb, nx, ny)
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in NSS_symmetries(na, nb, nx, ny, var_names):
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


"""
def NSS_vertices_from_BLM05(na, nb):
    \""" Returns vertices of NSS polytope in 2-choice case, where each choice of x has na outcomes,
     and each choice of y has nb outcomes. Calculated from results of BLM+05 \"""
    vertices = []
    ra = range(0, na)
    rb = range(0, nb)

    # First the deterministic vertices (i.e. the local vertices). Loop over deterministic functions f_a, f_b:
    for f_a in itertools.product(range(0, na - 1), repeat=2):
        for f_b in itertools.product(range(0, nb - 1), repeat=2):
            cor = []
            for a, b, x, y in itertools.product(ra, rb, (0, 1), (0, 1)):
                cor.append((f_a[x] == a) * (f_b[y] == b))
            vertices.append(cor)

    # Now the nonlocal vertices. See BLM+05 Eq. (6)
    """


def NSS_symmetries(na, nb, nx, ny, var_names):
    """ var_names should have length dim_NSS(na, nb, nx, ny) """
    symmetries = []  # array of strings

    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)
    ras_id = [ra, ] * nx  # the function {vals of x} -> {perms of ra} which is constantly {identity perm of ra}
    rbs_id = [rb, ] * ny

    # Recall that every permutation of [0, ..., m-1] can be written as a product of 2-cycles (n,n+1) for 0<=n<=m-2. So need only include those.
    # For permutations of values of a, note that we may allow these to depend on the value of x.
    # So loop over all functions ra_perms : {values of x} -> {2-cycle permutations of values of a}.

    # for ra_perms in itertools.product([list(ra)] + neighbourly_two_cycle_permutations(ra), repeat=nx):
    #     symmetries.append(permuted_NSS_var_names(ra_perms, rbs_id, rx, ry, var_names))
    # for rb_perms in itertools.product([list(rb)] + neighbourly_two_cycle_permutations(rb), repeat=ny):
    #     symmetries.append(permuted_NSS_var_names(ras_id, rb_perms, rx, ry, var_names))
    for ra_perm in neighbourly_two_cycle_permutations(ra):
        symmetries.append(permuted_NSS_var_names([ra_perm, ] * nx, rbs_id, rx, ry, var_names))
    for rb_perm in neighbourly_two_cycle_permutations(rb):
        symmetries.append(permuted_NSS_var_names(ras_id, [rb_perm, ] * ny, rx, ry, var_names))
    for rx_perm in neighbourly_two_cycle_permutations(rx):
        symmetries.append(permuted_NSS_var_names(ras_id, rbs_id, rx_perm, ry, var_names))
    for ry_perm in neighbourly_two_cycle_permutations(ry):
        symmetries.append(permuted_NSS_var_names(ras_id, rbs_id, rx, ry_perm, var_names))

    return symmetries
    # To test this function:
    # var_names2222 = ['a00', 'a01', 'b00', 'b01', 'ab0000', 'ab0001', 'ab0010', 'ab0011']
    # print('\n'.join(NSS_symmetries(2,2,2,2, var_names2222)))

    # In addition, for switching Alice and Bob around: +x2 +x3 +x0 +x1 +x4 +x6 +x5 +x7


def neighbourly_two_cycle_permutations(array):
    """ Returns a list of those permutations of `array' that are equal to the 2-cycle (n,n+1) for some n=0,...,len(array)-2. """
    perms = []
    for n in range(0, len(array) - 1):
        perm_array = [a for a in array]
        perm_array[n], perm_array[n + 1] = perm_array[n + 1], perm_array[n]
        perms.append(perm_array)
    return perms


def permuted_NSS_var_names(ras, rbs, rx, ry, var_names):
    """ ras and rab are assumed to be arrays of length nx and ny, respectively, such that
    ras[x] is a permutation of ra and rbs[y] is a permutation of rb. """
    # Below doesn't work because it doesn't take into account that e.g. p(a=0|x) = 1 - p(a=1|x)
    result = []  # length-NSS_dim array of strings representing algebraic expressions

    na = len(ras[0])
    nb = len(rbs[0])
    nx = len(rx)
    ny = len(ry)
    dim_NSS = polytope_utils.dim_NSS(na, nb, nx, ny)

    get_full_index = lambda a, b, x, y: a * nb * nx * ny + b * nx * ny + x * ny + y

    """
    # NSS-I
    for a, x in itertools.product(ra[0:na - 1], rx):
        result.append(var_names[get_NSS_I_index(a, x)])
    # NSS-II
    for b, y in itertools.product(rb[0:nb - 1], ry):
        result.append(var_names[get_NSS_II_index(b, y)])
    # NSS-III
    for a, b, x, y in itertools.product(ra[0:na - 1], rb[0:nb - 1], rx, ry):
        result.append(var_names[get_NSS_III_index(a, b, x, y)])

    return ' '.join(result)
    """

    NSS_to_full_matrix_but_weird = polytope_utils.construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny)

    # NSS-I
    for _as in [ras[x, 0:-1] for x in rx]:  # range over columns of ras, except the last one.
        for x in rx:
            a = _as[x]  # a as determined by the permutation singled out by x. TODO no these three lines are WRONG. Correct tomorrow. Also correct in NSS-II,III below!
            # Need to find expression for p(a|x) in terms of NSS coords.
            # p(a|x) = sum_b p(ab|x0), so need to find expression of p(ab|x0) in terms of NSS coords.
            # p(ab|x0) = get_full_index(a,b,x,0)-th row of NSS_to_full_matrix_but_weird, PLUS 1 if (a,b)=(na-1,nb-1)
            NSS_vector = np.zeros(dim_NSS)
            for b in range(0, nb):
                NSS_vector += NSS_to_full_matrix_but_weird[get_full_index(a, b, x, 0)]
            # If a=na-1, then since (a,b) has been (na-1,nb-1) once in the above loop, we have to correct for weirdness once, by adding 1.
            result.append(vector_to_expression(NSS_vector, var_names, a == na - 1))
    # NSS-II
    for y in ry:
        for b in rbs[y][0:-1]:
            NSS_vector = np.zeros(dim_NSS)
            for a in range(0, na):
                NSS_vector += NSS_to_full_matrix_but_weird[get_full_index(a, b, 0, y)]
            result.append(vector_to_expression(NSS_vector, var_names, b == nb - 1))
    # NSS-III
    for x, y in itertools.product(rx, ry):
        for b in rbs[y][0:-1]:
            for a in ras[x][0:-1]:
                NSS_vector = NSS_to_full_matrix_but_weird[get_full_index(a, b, x, y)]
                result.append(vector_to_expression(NSS_vector, var_names, (a, b) == (na - 1, nb - 1)))

    return ' '.join(result)

    # This function can be tested by e.g.:
    # var_names2222 = ['a00', 'a01', 'b00', 'b01', 'ab0000', 'ab0001', 'ab0010', 'ab0011']
    # print(permuted_NSS_var_names((1,0),(0,1),(0,1),(0,1), var_names2222))


def vector_to_expression(NSS_vector, var_names, add_one):
    """ If add_one=True, then add 1 to the expression, otherwise don't. """
    assert len(NSS_vector) == len(var_names)
    expr = '1' if add_one else ''
    for NSS_coord in range(0, len(NSS_vector)):
        if NSS_vector[NSS_coord] == 1:
            expr += '+' + var_names[NSS_coord]
        elif NSS_vector[NSS_coord] == -1:
            expr += '-' + var_names[NSS_coord]
        elif NSS_vector[NSS_coord] != 0:
            raise RuntimeError("Didn't think this would happen!")
    return expr


def readable_var_names(na, nb, nx, ny):
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


write_panda_input_for_nss(3, 3, 2, 2)
