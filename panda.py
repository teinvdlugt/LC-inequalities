import itertools
import numpy as np
import polytope_utils
import re


def write_panda_input_for_nss(na, nb, nx, ny, readable=False, filename=None):
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
        var_names = readable_var_names(na, nb, nx, ny)
    else:
        var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in NSS_symmetry_generators(na, nb, nx, ny, var_names):
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


def NSS_symmetry_generators(na, nb, nx, ny, var_names):
    """ var_names should have length dim_NSS(na, nb, nx, ny). 'generators' because it returns a generating subset of all symmetries
     (namely those given by 'neighbourly 2-cycles'. """
    symmetries = []  # array of strings

    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)
    ras = [ra, ] * nx  # the function {vals of x} -> {perms of ra} which is constantly {identity perm of ra}
    rbs = [rb, ] * ny

    # Recall that every permutation of [0, ..., m-1] can be written as a product of 2-cycles (n,n+1) for 0<=n<=m-2. So need only include those.
    # For permutations of values of a, note that we may allow these to depend on the value of x.
    # So loop over all functions ra_perms : {values of x} -> {2-cycle permutations of values of a}.

    fns_from_rx_to_nontriv_perms_of_ra = list(itertools.product([list(ra)] + neighbourly_two_cycle_permutations(ra), repeat=nx))[1:]  # [1:] because the first one is trivial
    fns_from_ry_to_nontriv_perms_of_rb = list(itertools.product([list(rb)] + neighbourly_two_cycle_permutations(rb), repeat=ny))[1:]  # (viz. it is the identity perm on ra, for all x)

    for ra_perms in fns_from_rx_to_nontriv_perms_of_ra:
        symmetries.append(var_perm_to_NSS_symm(ra_perms, rbs, rx, ry, var_names))
    for rb_perms in fns_from_ry_to_nontriv_perms_of_rb:
        symmetries.append(var_perm_to_NSS_symm(ras, rb_perms, rx, ry, var_names))
    for rx_perm in neighbourly_two_cycle_permutations(rx):
        symmetries.append(var_perm_to_NSS_symm(ras, rbs, rx_perm, ry, var_names))
    for ry_perm in neighbourly_two_cycle_permutations(ry):
        symmetries.append(var_perm_to_NSS_symm(ras, rbs, rx, ry_perm, var_names))

    # For uncontrolled permutations on a and b, use
    # for ra_perm in neighbourly_two_cycle_permutations(ra):
    #     symmetries.append(var_perm_to_NSS_symm([ra_perm, ] * nx, rbs, rx, ry, var_names))
    # for rb_perm in neighbourly_two_cycle_permutations(rb):
    #     symmetries.append(var_perm_to_NSS_symm(ras, [rb_perm, ] * ny, rx, ry, var_names))

    return symmetries
    # To test this function:
    # var_names2222 = ['a00', 'a01', 'b00', 'b01', 'ab0000', 'ab0001', 'ab0010', 'ab0011']
    # print('\n'.join(NSS_symmetry_generators(2,2,2,2, var_names2222)))

    # In addition, for switching Alice and Bob around: +x2 +x3 +x0 +x1 +x4 +x6 +x5 +x7


def identity_symmetry(var_names):
    result = ""
    for var in var_names:
        result += '+' + var + ' '
    return result[:-1]


def neighbourly_two_cycle_permutations(array):
    """ Returns a list of those permutations of `array' that are equal to the 2-cycle (n,n+1) for some n=0,...,len(array)-2. """
    perms = []
    for n in range(0, len(array) - 1):
        perm_array = [a for a in array]
        perm_array[n], perm_array[n + 1] = perm_array[n + 1], perm_array[n]
        perms.append(perm_array)
    return perms


def var_perm_to_NSS_symm(ra_perms, rb_perms, rx_perm, ry_perm, var_names):
    """ Returns a string of dim_NSS algebraic expressions separated by spaces. The i-th expression represents the
    i-th NSS coordinate of p^σ, which is the probability distribution defined by pullback p^σ = p∘σ, where σ is the
    permutation defined by ra_perms, rb_perms, rx_perm, ry_perm. [See p122]
    :param rx_perm: permutation of rx = range(0, nx)
    :param ry_perm: permutation of ry = range(0, ny)
    :param ra_perms: array of length nx, such that for each x, ra_perms[x] is a permutation of ra = range(0, na)
    :param rb_perms: array of length ny, such that for each y, rb_perms[y] is a permutation of rb = range(0, nb)
    """
    result = []  # length-NSS_dim array of strings, each representing an algebraic expression

    na = len(ra_perms[0])
    nb = len(rb_perms[0])
    nx = len(rx_perm)
    ny = len(ry_perm)
    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)

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

    # NSS-I coords of p^σ
    for a, x in itertools.product(ra[:-1], rx):
        _x = rx_perm[x]  # NOTE _x means σ(x), sim. for other variables in below code.
        _a = ra_perms[_x][a]  # See [p123] for explanation (in particular, why I don't choose to write ra_perms[x][a])
        # Now need to find expression of p^σ(a|x) = p(_a|_x) in terms of NSS-cpts of p.
        # p(_a|_x) = sum_b p(_a b|_x 0), so need to find expression of p(_a b|_x 0) in terms of NSS coords.
        # p(_a b|_x 0) = get_full_index(_a,b,_x,0)-th row of NSS_to_full_matrix_but_weird, PLUS 1 if (_a,b)=(na-1,nb-1)
        NSS_vector = np.zeros(dim_NSS)
        for b in range(0, nb):
            NSS_vector += NSS_to_full_matrix_but_weird[get_full_index(_a, b, _x, 0)]
        # If a=na-1, then since (a,b) has been (na-1,nb-1) once in the above loop, we have to correct for weirdness once, by adding 1.
        result.append(vector_to_expression(NSS_vector, var_names, _a == na - 1))
    # NSS-II
    for b, y in itertools.product(rb[:-1], ry):
        _y = ry_perm[y]  # NOTE _x means σ(x), sim. for other variables in below code.
        _b = rb_perms[_y][b]
        NSS_vector = np.zeros(dim_NSS)
        for a in range(0, na):  # Now summing over a
            NSS_vector += NSS_to_full_matrix_but_weird[get_full_index(a, _b, 0, _y)]
        result.append(vector_to_expression(NSS_vector, var_names, _b == nb - 1))
    # NSS-III
    for a, b, x, y in itertools.product(ra[:-1], rb[:-1], rx, ry):
        _x = rx_perm[x]
        _y = ry_perm[y]
        _a = ra_perms[_x][a]
        _b = rb_perms[_y][b]
        # Now need to express p^σ(ab|xy) = p(_a _b | _x _y) in terms of NSS cpts of p.
        NSS_vector = NSS_to_full_matrix_but_weird[get_full_index(_a, _b, _x, _y)]
        result.append(vector_to_expression(NSS_vector, var_names, (_a, _b) == (na - 1, nb - 1)))

    return ' '.join(result)

    """# NSS-I
    for _as in [ra_perms[x, 0:-1] for x in rx_perm]:  # range over columns of ras, except the last one.
        for x in rx_perm:
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
    for y in ry_perm:
        for b in rb_perms[y][0:-1]:
            NSS_vector = np.zeros(dim_NSS)
            for a in range(0, na):
                NSS_vector += NSS_to_full_matrix_but_weird[get_full_index(a, b, 0, y)]
            result.append(vector_to_expression(NSS_vector, var_names, b == nb - 1))
    # NSS-III
    for x, y in itertools.product(rx_perm, ry_perm):
        for b in rb_perms[y][0:-1]:
            for a in ra_perms[x][0:-1]:
                NSS_vector = NSS_to_full_matrix_but_weird[get_full_index(a, b, x, y)]
                result.append(vector_to_expression(NSS_vector, var_names, (a, b) == (na - 1, nb - 1)))

    return ' '.join(result)"""

    # This function can be tested by e.g.:
    # var_names2222 = ['a00', 'a01', 'b00', 'b01', 'ab0000', 'ab0001', 'ab0010', 'ab0011']
    # print(var_perm_to_NSS_symm((1,0),(0,1),(0,1),(0,1), var_names2222))


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
    # print(vertices_are_equivalent(3, 3, 2, 2, panda_vertex, BLM05_vertex, 5))
    # Gives True, so the vertices are equivalent.


def apply_symmetry(vertex, symmetry, var_names):
    """ Parses the symmetry and applies it to array, using the name mapping provided by var_names.
    :param vertex: float/int array that the symmetry should be applied to
    :param symmetry: string representing a symmetry. Guaranteed only to work if of the from as generated by var_perm_to_NSS_symm
    :param var_names: ordered list of strings. Should have same size as array.
    :return: a float/int array of same type as array.
    """
    new_vertex = []
    NSS_expressions = symmetry.split(' ')
    for NSS_expr in NSS_expressions:
        NSS_cpt = 0
        if NSS_expr[0] == '1':
            NSS_cpt += 1
            NSS_expr = NSS_expr[1:]
        # split the rest into individual var_names and +/- signs
        NSS_expr = re.split('([+-])', NSS_expr)[1:]  # gives e.g. ['+', 'x4', '-', 'x7'] if NSS_expr = '+x4-x7'
        assert len(NSS_expr) % 2 == 0

        for i in range(0, len(NSS_expr), 2):
            value_of_var_name = vertex[var_names.index(NSS_expr[i + 1])]
            if NSS_expr[i] == '+':
                NSS_cpt += value_of_var_name
            elif NSS_expr[i] == '-':
                NSS_cpt -= value_of_var_name
            else:
                raise ValueError('provided symmetry is of unsupported format')
        new_vertex.append(NSS_cpt)
    return new_vertex


def vertices_are_equivalent(na, nb, nx, ny, vertex1, vertex2, depth):
    """
    :param depth: NSS_symmetry_generators only returns the generators, not ALL symmetries. So this function has to loop
        through compositions of those generators. It only tries compositions of `depth' (or less) generators.
    """
    var_names = readable_var_names(na, nb, nx, ny)
    symm_generators = ['identity symmetry'] + NSS_symmetry_generators(na, nb, nx, ny, var_names)
    for generator_list in itertools.product(symm_generators, repeat=depth):  # loop through compositions of symmetries of length 20.
        # apply all symmetries in generator_list in succession
        new_vertex1 = [i for i in vertex1]
        for generator in generator_list:
            if generator != 'identity symmetry':
                new_vertex1 = apply_symmetry(new_vertex1, generator, var_names)
        if np.all(vertex2 == new_vertex1):
            return True
    return False

# write_panda_input_for_nss(3, 3, 2, 2, True)
