import itertools

import numpy as np
import symmetry_utils as symm
from utils import concatenate_bits, cart, B


def dim_NSS(na, nb, nx, ny):
    return nx * ny * (na - 1) * (nb - 1) + nx * (na - 1) + ny * (nb - 1)

def construct_full_to_NSS(na, nb, nx, ny):
    """
    Make and return matrix that converts shape (na*nb*nx*ny) vectors (full dimension) to shape (dim_nss(na, nb, nx, ny),)
    vectors (Collins-Gisin representation of no-signalling distributions; NSS stands for no-(superluminal-)signalling). ith row of the matrix represents ith NSS coordinate
    """

    full_dim = na * nb * nx * ny
    NSS_dim = dim_NSS(na, nb, nx, ny)
    matrix = np.zeros((NSS_dim, full_dim), dtype='int8')

    # Define ranges of variables in NSS rep
    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)

    get_index_in_full_rep = lambda a, b, x, y: a * nb * nx * ny + b * nx * ny + x * ny + y

    current_row = 0

    # NSS-I: p(a|x)
    for a, x in itertools.product(range(0, na - 1), rx):
        # p(a|x) = sum_b p(ab|x0)
        for b in rb:
            matrix[current_row][get_index_in_full_rep(a, b, x, 0)] = 1
        current_row += 1

    # NSS-II: p(b|y)
    for b, y in itertools.product(range(0, nb - 1), ry):
        # p(b|y) = sum_a p(ab|0y)
        for a in ra:
            matrix[current_row][get_index_in_full_rep(a, b, 0, y)] = 1
        current_row += 1

    # NS-III: p(ab|xy)
    for a, b, x, y in itertools.product(range(0, na - 1), range(0, nb - 1), rx, ry):
        matrix[current_row][get_index_in_full_rep(a, b, x, y)] = 1
        current_row += 1

    return matrix


def construct_full_to_NSS_h(na=8, nb=2, nx=4, ny=2):
    return np.block([[construct_full_to_NSS(na, nb, nx, ny), np.zeros((dim_NSS(na, nb, nx, ny), 1), dtype='int')],
                     [np.zeros(na * nb * nx * ny, dtype='int'), 1]])


def construct_NSS_to_full_h(na=8, nb=2, nx=4, ny=2):
    """
    See README.md.
    """
    # First create matrix of shape (nx*ny*na*nb, nx*ny*(na-1)(nb-1)+nx(na-1)+ny(nb-1)) that converts NSS coords
    #         {p(a|x), p(b|y), p(ab|xy)  |  a<na-1, b<nb-1, all x,y} =: NSS-I ∪ NSS-II ∪ NSS-III (see [p113])
    #     into 'weird' full coords
    #         {p(ab|xy)  |  (a,b) != (na-1,nb-1), all x,y} ∪ {p(na,nb|xy) - 1  |  all x,y}.

    # Define dimensions
    full_dim = na * nb * nx * ny
    NSS_dim = dim_NSS(na, nb, nx, ny)
    NSS_to_full_but_weird = np.zeros((full_dim, NSS_dim), dtype='int')

    # Define ranges of variables
    ra = range(0, na)
    rb = range(0, nb)
    rx = range(0, nx)
    ry = range(0, ny)

    # Useful functions to work with NSS coords
    NSS_II_offset = (na - 1) * nx  # the index at which NSS coords of type II [see p113] start within a NSS vector
    NSS_III_offset = NSS_II_offset + (nb - 1) * ny  # the index at which NSS coords of type III [see p113] start within a NSS vector
    get_NSS_I_index = lambda a, x: a * nx + x
    get_NSS_II_index = lambda b, y: NSS_II_offset + b * ny + y
    get_NSS_III_index = lambda a, b, x, y: NSS_III_offset + a * (nb - 1) * nx * ny + b * nx * ny + x * ny + y  # b only ranges over 0,...,nb-2

    # Loop over a,b,x,y, defining each p(ab|xy) in each iteration.
    current_row = 0
    for a, b, x, y in itertools.product(ra, rb, rx, ry):
        # See [p113] in Sketchbook for expressions in terms of NSS coords
        if a != na - 1 and b != nb - 1:
            # p(ab|xy) appears in NSS itself
            NSS_to_full_but_weird[current_row][get_NSS_III_index(a, b, x, y)] = 1
        elif a == na - 1 and b != nb - 1:
            # p(na-1 b|xy) = p(b|y) - sum_{_a < na-1} p(_a b|xy)
            NSS_to_full_but_weird[current_row][get_NSS_II_index(b, y)] = 1
            for _a in range(0, na - 1):
                NSS_to_full_but_weird[current_row][get_NSS_III_index(_a, b, x, y)] = -1
        elif a != na - 1 and b == nb - 1:
            # p(a nb-1|xy) = p(a|x) - sum_{_b < nb-1} p(a _b|xy)
            NSS_to_full_but_weird[current_row][get_NSS_I_index(a, x)] = 1
            for _b in range(0, nb - 1):
                NSS_to_full_but_weird[current_row][get_NSS_III_index(a, _b, x, y)] = -1
        elif a == na - 1 and b == nb - 1:
            # (For the current values of x,y), the rows corresponding to p(ab|xy) with (a,b)!=(na-1,nb-1) have already been filled.
            # So can use those to calculate the current row, which is -sum_{(a,b)!=(na-1,nb-1)} p(ab|xy)
            # NOTE that this isn't p(na-1,nb-1|xy) but instead it is p(na-1,nb-1|xy) - 1.
            for _a, _b in list(itertools.product(ra, rb))[0: na * nb - 1]:  # everything except (na-1, nb-1)
                NSS_to_full_but_weird[current_row] -= NSS_to_full_but_weird[_a * nb * nx * ny + _b * nx * ny + x * ny + y]
        else:
            print("something went wrong")
        current_row += 1

    return np.block([
        [NSS_to_full_but_weird, np.r_[np.zeros(na * nb * nx * ny - nx * ny), np.ones(nx * ny)].reshape((na * nb * nx * ny, 1))],
        [np.zeros(dim_NSS(na, nb, nx, ny)), 1]
    ]).astype('int8')



# LC1 stuff  ---- note, the order of the variables is always acbxyz (with b and c interchanged).
def dim_LC1(nabcxy):
    na1, na2, nc, nb, nx1, nx2, ny = nabcxy
    # For weak NSCO1 version: nx1 * (na1 - 1) + nx1 * nx2 * na1 * (na2 * nc - 1) + ny * (nb - 1) + nx1 * nx2 * ny * (na1 * na2 * nc - 1) * (nb - 1)
    # See [p131]:
    return nx1 * (na1 - 1) + ny * (nb - 1) + nx1 * ny * (na1 - 1) * (nb - 1) + nx1 * nx2 * na1 * (na2 * nc - 1) + nx1 * nx2 * ny * na1 * (na2 * nc - 1) * (nb - 1)


def construct_full_to_LC1():
    """
    Make and return matrix that converts length-128 vectors to length-80 vectors (LC1 rep of [p131])
    NOTE this function assumes the order first c, then b.
    """

    matrix = np.zeros((80, 128), dtype='int8')

    get_index_in_full_rep = lambda a1, a2, c, b, x1, x2, y: concatenate_bits(a1, a2, c, b, x1, x2, y)

    current_row = 0

    # NSCO1-I: p(a1=0|x1)
    for x1 in B:
        for a2, b, c in cart(B, B, B):
            matrix[current_row][get_index_in_full_rep(0, a2, c, b, x1, 0, 0)] = 1
        current_row += 1
    # NSCO1-II: p(b=0|y)
    for y in B:
        for a1, a2, c in cart(B, B, B):
            matrix[current_row][get_index_in_full_rep(a1, a2, c, 0, 0, 0, y)] = 1
        current_row += 1
    # NSCO1-III: p(a1=0 b=0|x1,y)
    for x1, y in cart(B, B):
        for a2, c in cart(B, B):
            matrix[current_row][get_index_in_full_rep(0, a2, c, 0, x1, 0, y)] = 1
        current_row += 1
    # NSCO1-IV: p(a1,a2,c|x1,x2), (a2,c) != (1,1)
    for a1, (a2, c), x1, x2 in cart(B, cart(B, B)[:-1], B, B):
        for b in B:
            matrix[current_row][get_index_in_full_rep(a1, a2, c, b, x1, x2, 0)] = 1
        current_row += 1
    # NSCO1-V: p(a1,a2,c,0|x1,x2,y), (a2,c) != (1,1)
    for a1, (a2, c), x1, x2, y in cart(B, cart(B, B)[:-1], B, B, B):
        matrix[current_row][get_index_in_full_rep(a1, a2, c, 0, x1, x2, y)] = 1
        current_row += 1

    return matrix


def construct_full_to_LC1_h():
    return np.block([
        [construct_full_to_LC1(), np.zeros((80, 1), dtype='int')],
        [np.zeros(128, dtype='int'), 1]
    ])


def construct_LC1_to_full_h():
    """
    See README.md.
    """

    # First create 'weird' matrix (see comments in construct_NSS_to_full_h):
    NSCO1_II_offset = 2
    NSCO1_III_offset = NSCO1_II_offset + 2
    NSCO1_IV_offset = NSCO1_III_offset + 4
    NSCO1_V_offset = NSCO1_IV_offset + 24

    NSCO1_I_index = lambda x1: x1
    NSCO1_II_index = lambda y: NSCO1_II_offset + y
    NSCO1_III_index = lambda x1, y: NSCO1_III_offset + x1 * 2 + y
    NSCO1_IV_index = lambda a1, a2, c, x1, x2: NSCO1_IV_offset + a1 * (2 * 2 - 1) * 2 * 2 + a2 * 2 * 2 * 2 + c * 2 * 2 + x1 * 2 + x2
    NSCO1_V_index = lambda a1, a2, c, x1, x2, y: NSCO1_V_offset + a1 * (2 * 2 - 1) * 1 * 8 + a2 * 2 * 8 + c * 8 + x1 * 4 + x2 * 2 + y

    matrix = np.zeros((128, 80), dtype='int8')   # <- the weird matrix
    current_row = 0  # Alternatively, can use concatenate_bits(a1, a2, c, b, x1, x2, y) in each iteration of the below for-loop.

    # See [p133]
    for a1, a2, c, b, x1, x2, y in cart(B, B, B, B, B, B, B):
        if b == 0:
            if (a2, c) != (1, 1):
                # case (i)  [cf. p133]
                matrix[current_row][NSCO1_V_index(a1, a2, c, x1, x2, y)] = 1
            elif a1 == 0:
                # case (ii)
                matrix[current_row][NSCO1_III_index(x1, y)] = 1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_V_index(0, _a2, _c, x1, x2, y)] = -1
            elif a1 == 1:
                # case (iii)
                matrix[current_row][NSCO1_II_index(y)] = 1
                matrix[current_row][NSCO1_III_index(x1, y)] = -1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_V_index(1, _a2, _c, x1, x2, y)] = -1
        elif b == 1:
            if (a2, c) != (1, 1):
                # case (iv)
                matrix[current_row][NSCO1_IV_index(a1, a2, c, x1, x2)] = 1
                matrix[current_row][NSCO1_V_index(a1, a2, c, x1, x2, y)] = -1
            elif a1 == 0:
                # case (v)
                matrix[current_row][NSCO1_I_index(x1)] = 1
                matrix[current_row][NSCO1_III_index(x1, y)] = -1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_IV_index(0, _a2, _c, x1, x2)] = -1
                    matrix[current_row][NSCO1_V_index(0, _a2, _c, x1, x2, y)] = 1
            elif a1 == 1:
                # case (vi): p(1111|x1x2y)-1
                matrix[current_row][NSCO1_I_index(x1)] = -1
                matrix[current_row][NSCO1_II_index(y)] = -1
                matrix[current_row][NSCO1_III_index(x1, y)] = 1
                for _a2, _c in cart(B, B)[:-1]:
                    matrix[current_row][NSCO1_IV_index(1, _a2, _c, x1, x2)] = -1
                    matrix[current_row][NSCO1_V_index(1, _a2, _c, x1, x2, y)] = 1

                # Let's check that this is the same as -1 * the sum of all previous p(a1a2cb|x1x2y), for the same x1x2y.
                check_row = np.zeros(80)
                for _a1, _a2, _c, _b in cart(B, B, B, B)[:-1]:
                    check_row -= matrix[concatenate_bits(_a1, _a2, _c, _b, x1, x2, y)]
                assert np.all(check_row == matrix[current_row])

        current_row += 1
    return np.block([
        [matrix, np.r_[np.zeros(120), np.ones(8)].reshape((128, 1)).astype('int')],
        [np.zeros(80, dtype='int'), 1]
    ])


def is_in_NSS(cor, na, nb, nx, ny, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length na*nb*nx*ny (or homogeneous version, len+1, also works) """
    if len(cor) == na * nb * nx * ny + 1:
        assert cor[-1] != 0
        cor = cor[:-1]  # actually doing the rescaling is unnecessary as it does not impact signalling
    else:
        assert len(cor) == na * nb * nx * ny

    # Probabilities >= 0
    if np.any(cor < -tol):
        return False

    cor_ab_xy = cor.reshape((na, nb, nx, ny))
    cor_b_xy = np.einsum('ijkl->jkl', cor_ab_xy)
    cor_a_xy = np.einsum('ijkl->ikl', cor_ab_xy)

    # Check for signalling A -> B
    for b, y in cart(range(0, nb), range(0, ny)):
        pby = cor_b_xy[b, 0, y]
        for x in range(1, nx):
            if abs(pby - cor_b_xy[b, x, y]) > tol:
                return False
    for a, x in cart(range(0, na), range(0, nx)):
        pax = cor_a_xy[a, x, 0]
        for y in range(1, ny):
            if abs(pax - cor_a_xy[a, x, y]) > tol:
                return False
    return True


def is_in_LC1(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    # First check if in NSS. This also checks if all probs are >=0.
    if not is_in_NSS(cor, 8, 2, 4, 2, tol):
        return False

    # Left to check: a1b independent of x2
    cor = cor.reshape((2,) * 7)
    cor_a1b_x1x2y = np.einsum('ijklmno->ilmno', cor)
    for a1, b, x1, y in itertools.product((0, 1), repeat=4):
        if abs(cor_a1b_x1x2y[a1, b, x1, 0, y] - cor_a1b_x1x2y[a1, b, x1, 1, y]) > tol:
            return False
    return True


def is_in_LC1st(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    # First check if in NSS. This also checks if all probs are >=0.
    if not is_in_NSS(cor, 8, 2, 4, 2, tol):
        return False

    # Left to check: a1 independent of x2
    cor = cor.reshape((2,) * 7)
    cor_a1_x1x2 = np.einsum('ijklmno->imno', cor)[:, :, :, 0]
    for a1, x1 in itertools.product((0, 1), repeat=2):
        if abs(cor_a1_x1x2[a1, x1, 0] - cor_a1_x1x2[a1, x1, 1]) > tol:
            return False
    return True


def is_in_LC2(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    swap_A1_A2_matrix = symm.full_perm_to_symm_h(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
    cor_swapped = (swap_A1_A2_matrix @ np.r_[cor, [1]])[:-1]
    return is_in_LC1(cor_swapped, tol)


def is_in_LC2st(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 128 or 129 """
    if len(cor) == 129:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 128

    swap_A1_A2_matrix = symm.full_perm_to_symm_h(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
    cor_swapped = (swap_A1_A2_matrix @ np.r_[cor, [1]])[:-1]
    return is_in_LC1st(cor_swapped, tol)


def one_hot_vector(size, position, dtype='int'):
    if position < 0:
        position += size
    return np.r_[np.zeros(position, dtype=dtype), [1], np.zeros(size - position - 1, dtype=dtype)]
