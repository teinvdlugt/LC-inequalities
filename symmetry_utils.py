import itertools
import re
import sys

import numpy as np

import panda
import utils
import vector_space_utils as vs
cart = vs.cart

B = (0, 1)


## NSS (i.e. pure Bell) stuff
def nss_symmetry_generators(na, nb, nx, ny, var_names):
    """ var_names should have length dim_NSS(na, nb, nx, ny). 'generators' because it returns a generating subset of all symmetries
     (namely those given by 'neighbourly 2-cycles'). """
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

    nontriv_fns_from_rx_to_perms_of_ra = list(itertools.product([list(ra)] + neighbourly_two_cycle_permutations(ra), repeat=nx))[1:]  # [1:] because the first one is trivial
    nontriv_fns_from_ry_to_perms_of_rb = list(itertools.product([list(rb)] + neighbourly_two_cycle_permutations(rb), repeat=ny))[1:]  # (viz. it is the identity perm on ra, for all x)

    for ra_perms in nontriv_fns_from_rx_to_perms_of_ra:
        symmetries.append(nss_var_perm_to_symm(ra_perms, rbs, rx, ry, var_names))
    for rb_perms in nontriv_fns_from_ry_to_perms_of_rb:
        symmetries.append(nss_var_perm_to_symm(ras, rb_perms, rx, ry, var_names))
    for rx_perm in neighbourly_two_cycle_permutations(rx):
        symmetries.append(nss_var_perm_to_symm(ras, rbs, rx_perm, ry, var_names))
    for ry_perm in neighbourly_two_cycle_permutations(ry):
        symmetries.append(nss_var_perm_to_symm(ras, rbs, rx, ry_perm, var_names))

    # For uncontrolled permutations on a and b, use
    # for ra_perm in neighbourly_two_cycle_permutations(ra):
    #     symmetries.append(nss_var_perm_to_symm([ra_perm, ] * nx, rbs, rx, ry, var_names))
    # for rb_perm in neighbourly_two_cycle_permutations(rb):
    #     symmetries.append(nss_var_perm_to_symm(ras, [rb_perm, ] * ny, rx, ry, var_names))

    return symmetries
    # To test this function:
    # var_names2222 = ['a00', 'a01', 'b00', 'b01', 'ab0000', 'ab0001', 'ab0010', 'ab0011']
    # print('\n'.join(nss_symmetry_generators(2,2,2,2, var_names2222)))

    # In addition, for switching Alice and Bob around: +x2 +x3 +x0 +x1 +x4 +x6 +x5 +x7


"""
def nss_var_perm_to_symm(ra_perms, rb_perms, rx_perm, ry_perm, var_names):
    \""" Returns a string of dim_NSS algebraic expressions separated by spaces. The i-th expression represents the
    i-th NSS coordinate of p^σ, which is the probability distribution defined by pullback p^σ = p∘σ, where σ is the
    permutation defined by ra_perms, rb_perms, rx_perm, ry_perm. [See p122]
    :param rx_perm: permutation of rx = range(0, nx)
    :param ry_perm: permutation of ry = range(0, ny)
    :param ra_perms: array of length nx, such that for each x, ra_perms[x] is a permutation of ra = range(0, na)
    :param rb_perms: array of length ny, such that for each y, rb_perms[y] is a permutation of rb = range(0, nb)
    \"""
    result = []  # length-NSS_dim array of strings, each representing an algebraic expression

    na = len(ra_perms[0])
    nb = len(rb_perms[0])
    nx = len(rx_perm)
    ny = len(ry_perm)
    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)

    dim_NSS = vs.dim_NSS(na, nb, nx, ny)

    get_full_index = lambda a, b, x, y: a * nb * nx * ny + b * nx * ny + x * ny + y

    NSS_to_full_matrix_but_weird = vs.construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny)

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
    # This function can be tested by e.g.:
    # var_names2222 = ['a00', 'a01', 'b00', 'b01', 'ab0000', 'ab0001', 'ab0010', 'ab0011']
    # print(nss_var_perm_to_symm((1,0),(0,1),(0,1),(0,1), var_names2222))
"""


def nss_vertices_are_equivalent_old(na, nb, nx, ny, vertex1, vertex2, depth):
    """
    :param depth: nss_symmetry_generators only returns the generators, not ALL symmetries. So this function has to loop
        through compositions of those generators. It only tries compositions of `depth' (or less) generators.
    """
    var_names = panda.nss_readable_var_names(na, nb, nx, ny)
    symm_generators = ['identity symmetry'] + nss_symmetry_generators(na, nb, nx, ny, var_names)
    for generator_list in itertools.product(symm_generators, repeat=depth):  # loop through compositions of symmetries of length 20.
        # apply all symmetries in generator_list in succession
        new_vertex1 = [i for i in vertex1]
        for generator in generator_list:
            if generator != 'identity symmetry':
                new_vertex1 = apply_symmetry_old(new_vertex1, generator, var_names)
        if np.all(vertex2 == new_vertex1):
            return True
    return False


def nss_var_perm_to_symm(perm, dtype='int8'):
    """ See documentation of nsco1_var_perm_to_symm. This is the same but in the NSS representation instead. See [p158]. """
    dim_NSS = 86

    symm_matrix = np.zeros((dim_NSS + 1, dim_NSS + 1))
    current_row = 0

    NSS_to_full_weird = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2)

    def add_standard_basis_vector_to_matrix_row(row_number, var_tuple):
        symm_matrix[row_number][:-1] += NSS_to_full_weird[
            vs.concatenate_bits(*var_tuple)]  # vs.concatenate_bits(*var_tuple) is the index of the standard basis vector characterised by var_tuple in the ordered standard ('full') basis.
        symm_matrix[row_number][-1] += var_tuple[:4] == (1, 1, 1, 1)

    # NSS-I coords of p^σ  NOTE NSS-I,II,III as defined on [p158]
    for (a1, a2, c), x1, x2 in cart(cart(B,B,B)[:-1], B, B):
        # Now what is p(_a1|_x1) in terms of NSCO1-cpts of p? p(_a1|_x1) = sum_{a2,c,b} p(_a1 a2 c b | _x1 0 0) and we know
        # what each p(_a1 a2 c b | _x1 0 0) is in NSCO1-cpts because we have NSCO1_to_full_weird.
        for b, y in vs.cart(B, (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSS-II
    for b, y in vs.cart((0,), B):
        for a1, a2, c, x1, x2 in vs.cart(B, B, B, (0,), (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSS-III
    for (a1, a2, c), b, x1, x2, y in vs.cart(cart(B, B, B)[:-1], (0,), B, B, B):
        add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    assert current_row == len(symm_matrix) - 1

    # The last row of the matrix should be: (0 0 ... 0 1)
    symm_matrix[-1][-1] = 1

    return symm_matrix.astype(dtype=dtype)


def apply_symmetry_old(vertex, symmetry, var_names):
    """ Parses the symmetry and applies it to array, using the name mapping provided by var_names.
    :param vertex: float/int array that the symmetry should be applied to
    :param symmetry: string representing a symmetry. Guaranteed only to work if of the from as generated by nss_var_perm_to_symm
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


## NSCO1 stuff
# NOTE Always stick to the order a1 a2 c b x1 x2 y, so c first, then b. (Esp. for NSCO1-V)
def nsco1_symm_generators():
    return np.array([
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, (x1 + 1) % 2, x2, y)),  # x1 -> x1 + 1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: ((a1 + x1) % 2, a2, c, b, x1, x2, y)),  # a1 -> a1 + x1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, (x2 + x1 * a1) % 2, y)),  # x2 -> x2 + x1*a1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, (a2 + x1 * a1 * x2) % 2, c, b, x1, x2, y)),  # a2 -> a2 + x1*a1*x2
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, (c + x1 * a1 * x2 * a2) % 2, b, x1, x2, y)),  # c -> c + x1*a1*x2*a2
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, x2, (y + 1) % 2)),  # y  -> y  + 1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, (b + y) % 2, x1, x2, y))  # b  -> b  + y
    ])

def nsco1_symm_generators_in_nss_coords():
    return np.array([
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, (x1 + 1) % 2, x2, y)),  # x1 -> x1 + 1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: ((a1 + x1) % 2, a2, c, b, x1, x2, y)),  # a1 -> a1 + x1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, (x2 + x1 * a1) % 2, y)),  # x2 -> x2 + x1*a1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, (a2 + x1 * a1 * x2) % 2, c, b, x1, x2, y)),  # a2 -> a2 + x1*a1*x2
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, (c + x1 * a1 * x2 * a2) % 2, b, x1, x2, y)),  # c -> c + x1*a1*x2*a2
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, x2, (y + 1) % 2)),  # y  -> y  + 1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, (b + y) % 2, x1, x2, y))  # b  -> b  + y
    ])


def nsco1_var_perm_to_symm(perm, dtype='int8'):
    """
    :param perm: a function with 7 binary arguments and one binary output.
    :return: the affine map p -> p^σ (with σ := perm), encoded as a shape-(dim+1,dim+1) matrix in the following way:
      - for i != n+1, if the i-th coordinate of p^σ can be expressed as a1x1+...+anxn+c, with ai,c ∊ R and p = (x1 x2 ... xn)^T,
                    then the i-th row of the returned matrix is (a1 a2 ... an c).
      - the last ((n+1)-st) row is (0 0 ... 0 1).#
      This means that you can multiply any row (x1 x2 ... xn d) representing a vector (x1/d x2/d ... xn/d)^T with the matrix
      from the left to get the row representation of the symmetry applied to the vector. Namely:
        row:                    (x1  x2  x3  ...  xn  d)^T, with d the denominator
        symmetry i-th row:      (a1  a2  a3  ...  an  c), with c the constant in the affine sum  (for i != n+1)
        symmetry (n+1)-st row:  (0   0   0   ...  0   1)
           --> i-th elt of symmetry @ row (i!=n+1):  a1x1 + a2x2 + ... + anxn + dc  (for i != n+1)
               (n+1)-st element of symmetry @ row:   d
           --> so symmetry @ row represents the n-dim vector with elements of the form  a1(x1/d) + a2(x2/d) + ... + an(xn/d) + c
               exactly what we want!
    TODO think about: does this also work if row represents an inequality rather than a vertex? PANDA code seems to suggest so.
    """
    dim_NSCO1 = 80

    symm_matrix = np.zeros((dim_NSCO1 + 1, dim_NSCO1 + 1))
    current_row = 0

    NSCO1_to_full_weird = vs.construct_NSCO1_to_full_weird_matrix()

    def add_standard_basis_vector_to_matrix_row(row_number, var_tuple):
        symm_matrix[row_number][:-1] += NSCO1_to_full_weird[
            vs.concatenate_bits(*var_tuple)]  # vs.concatenate_bits(*var_tuple) is the index of the standard basis vector characterised by var_tuple in the ordered standard ('full') basis.
        symm_matrix[row_number][-1] += var_tuple[:4] == (1, 1, 1, 1)

    # NSCO1-I coords of p^σ
    for a1, x1 in vs.cart((0,), B):
        # Now what is p(_a1|_x1) in terms of NSCO1-cpts of p? p(_a1|_x1) = sum_{a2,c,b} p(_a1 a2 c b | _x1 0 0) and we know
        # what each p(_a1 a2 c b | _x1 0 0) is in NSCO1-cpts because we have NSCO1_to_full_weird.
        for a2, c, b, x2, y in vs.cart(B, B, B, (0,), (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-II
    for b, y in vs.cart((0,), B):
        for a1, a2, c, x1, x2 in vs.cart(B, B, B, (0,), (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-III
    for a1, b, x1, y in vs.cart((0,), (0,), B, B):
        for a2, c, x2 in vs.cart(B, B, (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-IV
    for a1, (a2, c), x1, x2 in vs.cart(B, vs.cart(B, B)[:-1], B, B):
        for b, y in vs.cart(B, (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-V
    for a1, (a2, c), b, x1, x2, y in vs.cart(B, vs.cart(B, B)[:-1], (0,), B, B, B):
        add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    assert current_row == len(symm_matrix) - 1

    # The last row of the matrix should be: (0 0 ... 0 1)
    symm_matrix[-1][-1] = 1

    return symm_matrix.astype(dtype=dtype)


def symm_matrix_to_string(symm_matrix, var_names):
    """
    :param symm_matrix: Should be square (dim+1,dim+1), and of the form as explained in nsco1_var_perm_to_symm()
    :param var_names:   list of strings of length dim.
    :return: a string containing dim=len(var_names) space-separated affine expressions
    """
    assert len(symm_matrix.shape) == 2 and symm_matrix.shape[0] == symm_matrix.shape[1] and symm_matrix.shape[0] == len(var_names) + 1

    string = ''
    for row in symm_matrix[:-1]:
        string += row_to_expression(row, var_names)
        string += ' '
    return string[:-1]  # Remove the last space


## LC stuff
def lc_symm_generators():
    return np.array([
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, (x1 + 1) % 2, x2, y)),  # x1 -> x1 + 1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: ((a1 + x1) % 2, a2, c, b, x1, x2, y)),  # a1 -> a1 + x1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, (x2 + 1) % 2, y)),  # x2 -> x2 + 1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, (a2 + x2) % 2, c, b, x1, x2, y)),  # a2 -> a2 + x2
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, (c + x1 * a1 * x2 * a2) % 2, b, x1, x2, y)),  # c -> c + x1*a1*x2*a2
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, x2, (y + 1) % 2)),  # y  -> y  + 1
        nss_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, (b + y) % 2, x1, x2, y))  # b  -> b  + y
    ])


## General stuff
def neighbourly_two_cycle_permutations(array):
    """ Returns a list of those permutations of `array' that are equal to the 2-cycle (n,n+1) for some n=0,...,len(array)-2. """
    perms = []
    for n in range(0, len(array) - 1):
        perm_array = [a for a in array]
        perm_array[n], perm_array[n + 1] = perm_array[n + 1], perm_array[n]
        perms.append(perm_array)
    return perms


def row_to_expression(row, var_names):
    """ If row=(a1, a2, c) and var_names=("x1", "x2"), then this function returns "c+a1x1+a2x2"  """
    assert len(row) == 1 + len(var_names)
    expr = '' if row[-1] == 0 else str(row[-1])
    for i in range(0, len(var_names)):
        if row[i] == 1:
            expr += '+' + var_names[i]
        elif row[i] == -1:
            expr += '-' + var_names[i]
        elif row[i] > 0:
            expr += '+' + str(row[i]) + var_names[i]
        elif row[i] < 0:
            expr += str(row[i]) + var_names[i]
    return expr


def get_class(row, symmetry_generators, console_output=True):
    """ Returns the orbit of 'row' under the group generated by 'symmetry_generators'.
    :param row: shape-(dim+1,) array. The last element represent the denominator.
    :param symmetry_generators: array of shape-(dim+1,dim+1) arrays, each representing an affine map as described in nsco1_var_perm_to_symm.
    :return: a list of length-(dim+1) lists: the equivalence class of row under the symmetry group spanned by symmetry_generators.
    """
    # Numpy data structure messes things up (e.g. when asking 'if new_node not in result') so let's do as much as possible with pure Python lists.
    result = [list(row)]

    # Below is like depth-first-search through the tree where each node corresponds
    # to a vector and splitting at a node corresponds to application of the different symmetry generators.
    # The stop looking in a particular branch (i.e. we 'create a leaf') if we find that no matter what
    # symmetry generator we apply, we get nothing new.
    # (Inspired by the PANDA source code algorithm_classes.cpp, getClass() function)
    nodes_to_explore = [row]  # 'row' is the root of our tree
    while len(nodes_to_explore) != 0:
        if console_output:
            sys.stdout.write('\rConstructing equivalence class... nodes to explore: %i, class size: %i' % (len(nodes_to_explore), len(result)))
        current_node = nodes_to_explore.pop()  # the current tree node that we're looking at
        for symm in symmetry_generators:  # loop over the branches splitting from this node
            new_node = list(np.matmul(symm, current_node))
            if new_node not in result:  # we found a new node!
                result.append(new_node)
                nodes_to_explore.append(new_node)
    if console_output:
        sys.stdout.write('\rConstructing equivalence class... done')
    # sys.stdout.flush()
    return result


def get_class_representative(row, symmetry_generators, console_output=True):
    # Return the lexicographically biggest member of the equivalence class of row
    the_class = get_class(row, symmetry_generators, console_output)
    return len(the_class), max(the_class)


if __name__ == '__main__':
    # symms_a2_controlled_by_x2, count_x2 = nsco1_symmetries_old(False, True)
    # print('total', symms_a2_controlled_by_x2.shape[0], 'symmetries;', len(np.argwhere(count_x2 > 1)), ' symmetries which had at least 1 duplicate')
    # symms_a2_controlled_by_x1x2, count_x1x2 = nsco1_symmetries_old(False, False)
    # print('total', symms_a2_controlled_by_x1x2.shape[0], 'symmetries;', len(np.argwhere(count_x1x2 > 1)), ' symmetries which had at least 1 duplicate')
    # print('the symmetries generated are', 'equal' if np.all(symms_a2_controlled_by_x2 == symms_a2_controlled_by_x1x2) else 'unequal')

    # perm = lambda a1, a2, c, b, x1, x2, y: (a1 + 1, a2, c, b, x1, x2, (y + 1) % 2)
    # nsco1_var_perm_to_symm(perm)

    # To test get_class and get_class_representative:
    row = list(map(int,
                   "  2  2  1  1  1  1  1  1  2  1  1  1  0  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  1  1  1  1  1  1  1  0  0  0  0  0  0  0  0  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  0  2".split()))
    symm_generators = nsco1_symm_generators()
    equiv_class = get_class(row, symm_generators)
    print('Class is', len(equiv_class), 'big.')
    equiv_row = list(symm_generators[5] @ row)
    assert equiv_row in equiv_class
    assert list(symm_generators[5] @ symm_generators[6] @ row) in equiv_class
    assert get_class_representative(equiv_row, symm_generators) == row
    print('Success!')
