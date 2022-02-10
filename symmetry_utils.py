import itertools
import re

import numpy as np

import panda
import utils
import vector_space_utils as vs
from vector_space_utils import cart

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


def nss_var_perm_to_symm(ra_perms, rb_perms, rx_perm, ry_perm, var_names):
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


def nss_vertices_are_equivalent(na, nb, nx, ny, vertex1, vertex2, depth):
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
                new_vertex1 = apply_symmetry(new_vertex1, generator, var_names)
        if np.all(vertex2 == new_vertex1):
            return True
    return False


## NSCO1 stuff
# NOTE Always stick to the order a1 a2 c b x1 x2 y, so c first, then b. (Esp. for NSCO1-V)

def nsco1_symmetries_old(only_output_generators=True, a2_only_ctrled_by_x2=False, c_ctrled=True):
    """
    var_names should have length dim_NSCO1. If just_output_generators=True, it returns a generating subset of all symmetries (namely
    those where only one of the 7 random variables is permuted). If False, it returns all symmetries.
    """
    symmetries = []  # array of (80,81)-shape matrices. See nsco1_var_perm_to_symm_old docs for what they represent.

    # In the binary case, there are just two permutations: identity and bitflip.
    # For permutations of values of a1, a2, b, note that we may allow these to depend on the value of x1, x2, y, respectively.
    # TODO can the permutation of a2 also depend on x1? And that of c on x1 and x2?
    # So loop over functions ra1_perms : {values of x} -> {permutations of values of a}, i.e. B -> ((0,1), (1,0))

    perms_of_B = ((0, 1), (1, 0))
    fns_from_B_to_perms_of_B = cart(perms_of_B, perms_of_B)
    # The following is for use when only_output_generators==True. In that case we only need two of the four 'controlled symmetries' of a1:
    # one in which the permutation applied to a1 is different for different x1 fns_from_B_to_perms_of_B[2], and one in which the permutation
    # is always the bitflip (fns_from_B_to_perms_of_B[3]).
    # The one which maps all x1 to the identity permutation (fns_from_B_to_perms_of_B[0]) is not interesting, and the other one which maps
    # different x1 to different permutations of a1 (fns_from_B_to_perms_of_B[1]) is generated by a bitflip permutation of x1
    # followed by fns_from_B_to_perms_of_B[2].
    less_fns_from_B_to_perms_of_B = (((1, 0), (1, 0)), ((0, 1), (1, 0)))  # this order is nicer than fns_from_B_to_perms_of_B[2:]
    Bs = (B, B)  # the 'trivial controlled permutation' of B
    # Now consider permutations controlled by 2 parameters instead of 1 (i.e. perm of a2,c may be controlled by x1 AND x2):
    fns_from_4_to_perms_of_B = cart(perms_of_B, perms_of_B, perms_of_B, perms_of_B)
    B2_ctrled_B_perms = [lambda x1, x2, fn2=fn: fn2[x1 * 2 + x2] for fn in fns_from_4_to_perms_of_B]  # more intuitive data structure
    B2_ctrled_B_perms_but_actually_dep_on_x1_is_trivial = [lambda x1, x2, fn2=fn: fn2[x2] for fn in
                                                           fns_from_B_to_perms_of_B]  # 'fn2=fn' makes a variable local to the lambda and which is fixed from the definition of the lambda onwards
    # And a generating subset of fns_from_B2_to_perms_of_B:  [see p136]
    less_B2_ctrled_B_perms = [lambda x1, x2: (1, 0),  # c <-> c + 1
                              lambda x1, x2: perms_of_B[x1],  # c <-> c + x1
                              lambda x1, x2: perms_of_B[x2],  # c <-> c + x2
                              lambda x1, x2: perms_of_B[x1 * x2]]  # c <-> c + x1*x2
    triv_B2_controlled_B_perm = lambda x1, x2: (0, 1)
    const_B2_ctrled_B_perms = [lambda x1, x2: (0, 1), lambda x1, x2: (1, 0)]  # controlled permutations that are actually uncontrolled

    if only_output_generators:
        for B_perms in less_fns_from_B_to_perms_of_B:
            symmetries.append(nsco1_var_perm_to_symm_old(B_perms, triv_B2_controlled_B_perm, triv_B2_controlled_B_perm, Bs, B, B, B))  # controlled permutation of a1
            symmetries.append(nsco1_var_perm_to_symm_old(Bs, triv_B2_controlled_B_perm, triv_B2_controlled_B_perm, B_perms, B, B, B))  # controlled permutation of b
        for B2_controlled_B_perm in less_B2_ctrled_B_perms:
            symmetries.append(nsco1_var_perm_to_symm_old(Bs, B2_controlled_B_perm, triv_B2_controlled_B_perm, Bs, B, B, B))  # controlled permutation of a2
            symmetries.append(nsco1_var_perm_to_symm_old(Bs, triv_B2_controlled_B_perm, B2_controlled_B_perm, Bs, B, B, B))  # controlled permutation of c
        # For the x1, x2, y permutations, there's only one nontrivial permutation: (1,0)
        symmetries.append(nsco1_var_perm_to_symm_old(Bs, triv_B2_controlled_B_perm, triv_B2_controlled_B_perm, Bs, (1, 0), B, B))  # permutation of x1
        symmetries.append(nsco1_var_perm_to_symm_old(Bs, triv_B2_controlled_B_perm, triv_B2_controlled_B_perm, Bs, B, (1, 0), B))  # permutation of x2
        symmetries.append(nsco1_var_perm_to_symm_old(Bs, triv_B2_controlled_B_perm, triv_B2_controlled_B_perm, Bs, B, B, (1, 0)))  # permutation of y
    else:
        for a1_perms, a2_ctrled_perm, c_ctrled_perm, b_perms, x1_perm, x2_perm, y_perm in \
                cart(fns_from_B_to_perms_of_B, B2_ctrled_B_perms_but_actually_dep_on_x1_is_trivial if a2_only_ctrled_by_x2 else B2_ctrled_B_perms,
                     B2_ctrled_B_perms if c_ctrled else const_B2_ctrled_B_perms, fns_from_B_to_perms_of_B, perms_of_B, perms_of_B, perms_of_B):
            # cart(((B,B),((1,0),(1,0))), )
            symmetries.append(nsco1_var_perm_to_symm_old(a1_perms, a2_ctrled_perm, c_ctrled_perm, b_perms, x1_perm, x2_perm, y_perm))
        # Eliminate duplicates:
        # symmetries = utils.eliminate_duplicate_rows(symmetries)  # if you don't want them sorted too
        print(len(symmetries), 'symmetries before eliminating duplicates')
        symmetries, counts = np.unique(np.array(symmetries), axis=0, return_counts=True)
        print(len(symmetries), 'symmetries after eliminating duplicates')

    # For uncontrolled permutations on a1, a2 and b, simply redefine
    # less_fns_from_B_to_perms_of_B = fns_from_B_to_perms_of_B[3:4]  # a one-element array containing just the function that maps all x1 to the bitflip perm on a1.

    return np.array(symmetries)


def nsco1_var_perm_to_symm_old(a1_perms, a2_ctrled_perm, c_ctrled_perm, b_perms, x1_perm, x2_perm, y_perm, dtype='int8'):
    """ Returns a shape-(dim_NSCO1,dim_NSCO1+1) matrix. The i-th row, having length 81, represents the
    i-th NSCO1 coordinate of p^σ expressed affine map on NSCO1 coordinates, where p^σ which is the probability distribution
    defined by pullback p^σ = p∘σ, where σ is the permutation defined by a1_perms, a2_ctrled_perm, c_ctrled_perm, b_perms, x1_perm, x2_perm,
    y_perm. [See p122]  I.e. the first 80 entries in each row represent coefficients for NSCO1 vectors, while the last (81st) entry
    is 1 iff we should add 1 to the expression (this adding 1 makes the symmetry an affine map, as opposed to a linear one).
    :param a1_perms: size-2 array such that a1_perms[x1] is a permutation of the values of a1 (i.e. a permutation of B).
    :param a2_ctrled_perm: a function from two binary variables ('x1,x2') to a permutation of B
    :param c_ctrled_perm: a function from two binary variables ('x1,x2') to a permutation of B
    :param b_perms: sim. to a1_perms.
    :param x1_perm: sim. to c_perm.
    :param x2_perm: sim.
    :param y_perm: sim.
    """
    dim_NSCO1 = 80

    symm_matrix = np.zeros((dim_NSCO1, dim_NSCO1 + 1))  # each row has length 81: the laste entry encodes whether we should add 1 to the
    # expression or not (after all, symmetries are affine maps, not just linear maps).
    current_row = 0

    get_full_index = lambda a1, a2, c, b, x1, x2, y: vs.concatenate_bits(a1, a2, c, b, x1, x2, y)

    NSCO1_to_full_weird = vs.construct_NSCO1_to_full_weird_matrix()

    # NSCO1-I coords of p^σ
    for a1, x1 in cart((0,), B):
        _x1 = x1_perm[x1]  # NOTE As before (for NSS), _x1 means σ(x1); sim. for other variables in below code.
        _a1 = a1_perms[x1][a1]
        # Now what is p(_a1|_x1) in terms of NSCO1-cpts of p? p(_a1|_x1) = sum_{a2,c,b} p(_a1 a2 c b | _x1 0 0) and we know
        # what each p(_a1 a2 c b | _x1 0 0) is in NSCO1-cpts because we have NSCO1_to_full_weird.
        for a2, c, b in cart(B, B, B):
            symm_matrix[current_row][:-1] += NSCO1_to_full_weird[get_full_index(_a1, a2, c, b, _x1, 0, 0)]
        # If _a1==1, then since (_a1,a2,c,b) has been (1,1,1,1) once in the above loop, we have to correct for weirdness once, by adding 1.
        symm_matrix[current_row][-1] += _a1 == 1
        current_row += 1
    # NSCO1-II
    for b, y in cart((0,), B):
        _y = y_perm[y]
        _b = b_perms[y][b]
        for a1, a2, c in cart(B, B, B):
            symm_matrix[current_row][:-1] += NSCO1_to_full_weird[get_full_index(a1, a2, c, _b, 0, 0, _y)]
        symm_matrix[current_row][-1] += _b == 1
        current_row += 1
    # NSCO1-III
    for a1, b, x1, y in cart((0,), (0,), B, B):
        _x1, _y = x1_perm[x1], y_perm[y]
        _a1, _b = a1_perms[x1][a1], b_perms[y][b]
        for a2, c in cart(B, B):
            symm_matrix[current_row][:-1] += NSCO1_to_full_weird[get_full_index(_a1, a2, c, _b, _x1, 0, _y)]
        symm_matrix[current_row][-1] += (_a1, _b) == (1, 1)
        current_row += 1
    # NSCO1-IV
    for a1, (a2, c), x1, x2 in cart(B, cart(B, B)[:-1], B, B):
        _x1, _x2 = x1_perm[x1], x2_perm[x2]
        _a1, _a2, _c = a1_perms[x1][a1], a2_ctrled_perm(x1, x2)[a2], c_ctrled_perm(x1, x2)[c]
        for b in B:
            symm_matrix[current_row][:-1] += NSCO1_to_full_weird[get_full_index(_a1, _a2, _c, b, _x1, _x2, 0)]
        symm_matrix[current_row][-1] += (_a1, _a2, _c) == (1, 1, 1)
        current_row += 1
    # NSCO1-V
    for a1, (a2, c), b, x1, x2, y in cart(B, cart(B, B)[:-1], (0,), B, B, B):
        _x1, _x2, _y = x1_perm[x1], x2_perm[x2], y_perm[y]
        _a1, _a2, _c, _b = a1_perms[x1][a1], a2_ctrled_perm(x1, x2)[a2], c_ctrled_perm(x1, x2)[c], b_perms[y][b]
        symm_matrix[current_row][:-1] = NSCO1_to_full_weird[get_full_index(_a1, _a2, _c, _b, _x1, _x2, _y)]
        symm_matrix[current_row][-1] += (_a1, _a2, _c, _b) == (1, 1, 1, 1)
        current_row += 1

    return symm_matrix.astype(dtype=dtype)


def nsco1_symm_generators():
    symmetries = [
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, (x1 + 1) % 2, x2, y)),  # x1 -> x1 + 1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: ((a1 + x1) % 2, a2, c, b, x1, x2, y)),  # a1 -> a1 + x1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, (x2 + x1 * a1) % 2, y)),  # x2 -> x2 + x1*a1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, (a2 + x1 * a1 * x2) % 2, c, b, x1, x2, y)),  # a2 -> a2 + x1*a1*x2
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, (c + x1 * a1 * x2 * a2) % 2, b, x1, x2, y)),  # c -> c + x1*a1*x2*a2
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, b, x1, x2, (y + 1) % 2)),  # y  -> y  + 1
        nsco1_var_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a1, a2, c, (b + y) % 2, x1, x2, y))  # b  -> b  + y
    ]

    return np.array(symmetries)


def nsco1_var_perm_to_symm(perm, dtype='int8'):
    """
    :param perm: a function with 7 binary arguments and one binary output.
    :return: the symmetry p -> p^σ (with σ := perm), encoded in the same way as in the output of nsco1_var_perm_to_symm_old().
    """
    dim_NSCO1 = 80

    symm_matrix = np.zeros((dim_NSCO1, dim_NSCO1 + 1))  # each row has length 81: the laste entry encodes whether we should add 1 to the
    # expression or not (after all, symmetries are affine maps, not just linear maps).
    current_row = 0

    NSCO1_to_full_weird = vs.construct_NSCO1_to_full_weird_matrix()

    def add_standard_basis_vector_to_matrix_row(row_number, var_tuple):
        symm_matrix[row_number][:-1] += NSCO1_to_full_weird[
            vs.concatenate_bits(*var_tuple)]  # vs.concatenate_bits(*var_tuple) is the index of the standard basis vector characterised by var_tuple in the ordered standard ('full') basis.
        symm_matrix[row_number][-1] += var_tuple[:4] == (1, 1, 1, 1)

    # NSCO1-I coords of p^σ
    for a1, x1 in cart((0,), B):
        # Now what is p(_a1|_x1) in terms of NSCO1-cpts of p? p(_a1|_x1) = sum_{a2,c,b} p(_a1 a2 c b | _x1 0 0) and we know
        # what each p(_a1 a2 c b | _x1 0 0) is in NSCO1-cpts because we have NSCO1_to_full_weird.
        for a2, c, b, x2, y in cart(B, B, B, (0,), (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-II
    for b, y in cart((0,), B):
        for a1, a2, c, x1, x2 in cart(B, B, B, (0,), (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-III
    for a1, b, x1, y in cart((0,), (0,), B, B):
        for a2, c, x2 in cart(B, B, (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-IV
    for a1, (a2, c), x1, x2 in cart(B, cart(B, B)[:-1], B, B):
        for b, y in cart(B, (0,)):
            add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1
    # NSCO1-V
    for a1, (a2, c), b, x1, x2, y in cart(B, cart(B, B)[:-1], (0,), B, B, B):
        add_standard_basis_vector_to_matrix_row(current_row, perm(a1, a2, c, b, x1, x2, y))
        current_row += 1

    return symm_matrix.astype(dtype=dtype)


def symm_matrix_to_string(symm_matrix, var_names):
    assert len(symm_matrix.shape) and symm_matrix.shape[0] + 1 == symm_matrix.shape[1]

    string = ''
    for row in symm_matrix:
        string += vector_to_expression(row[:-1], var_names, row[-1])
        string += ' '
    return string[:-1]  # Remove the last space


## General stuff
def neighbourly_two_cycle_permutations(array):
    """ Returns a list of those permutations of `array' that are equal to the 2-cycle (n,n+1) for some n=0,...,len(array)-2. """
    perms = []
    for n in range(0, len(array) - 1):
        perm_array = [a for a in array]
        perm_array[n], perm_array[n + 1] = perm_array[n + 1], perm_array[n]
        perms.append(perm_array)
    return perms


def vector_to_expression(vector, var_names, add_one):
    """ If add_one=True, then add 1 to the expression, otherwise don't. """
    assert len(vector) == len(var_names)
    expr = '1' if add_one else ''
    for NSS_coord in range(0, len(vector)):
        if vector[NSS_coord] == 1:
            expr += '+' + var_names[NSS_coord]
        elif vector[NSS_coord] == -1:
            expr += '-' + var_names[NSS_coord]
        elif vector[NSS_coord] != 0:
            raise RuntimeError("Didn't think this would happen!")
    return expr


def apply_symmetry(vertex, symmetry, var_names):
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


if __name__ == '__main__':
    # symms_a2_controlled_by_x2, count_x2 = nsco1_symmetries_old(False, True)
    # print('total', symms_a2_controlled_by_x2.shape[0], 'symmetries;', len(np.argwhere(count_x2 > 1)), ' symmetries which had at least 1 duplicate')
    # symms_a2_controlled_by_x1x2, count_x1x2 = nsco1_symmetries_old(False, False)
    # print('total', symms_a2_controlled_by_x1x2.shape[0], 'symmetries;', len(np.argwhere(count_x1x2 > 1)), ' symmetries which had at least 1 duplicate')
    # print('the symmetries generated are', 'equal' if np.all(symms_a2_controlled_by_x2 == symms_a2_controlled_by_x1x2) else 'unequal')

    perm = lambda a1, a2, c, b, x1, x2, y: (a1 + 1, a2, c, b, x1, x2, (y + 1) % 2)
    nsco1_var_perm_to_symm(perm)
