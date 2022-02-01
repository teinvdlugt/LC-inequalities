import itertools

import numpy as np
from scipy.optimize import linprog


def in_hull(vertices, point, tol=1e-8):
    """ Decide if point is in the convex hull of vertices. See [p71] of notebook.
    tol: tolerance value. Is passed on to scipy. Default value for scipy is 1e-8.
    Based on https://stackoverflow.com/a/43564754/4129739
    scipy.optimize.linprog docs: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html
    Alternatively with MATLAB: https://uk.mathworks.com/matlabcentral/fileexchange/10226-inhull
    or https://github.com/coin-or/CyLP"""
    n = len(vertices)  # number of vertices.
    c = np.zeros(n)
    A = np.r_[vertices.T, np.ones(
        (1, n))]  # Transpose 'vertices', because in numpy they are provided as a column vector of row vectors
    b = np.r_[point, np.ones(1)]
    lp = linprog(c, A_eq=A, b_eq=b, options={'tol': tol})
    return lp.success  # solution of optimisation problem is not relevant - only if a solution was found


"""
# Let's test with easy polytope
vertices = np.array([[0, 0, 0],
                     [1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
point_that_is_in = np.array([1 / 5, 1 / 5, 1 / 5])
point_on_boundary = np.array(
    [1 / 3 + 1e-8, 1 / 3, 1 / 3])  # For 1/3+ 1e-9, it thinks it's inside the polytope. That corresponds to tol=1e-8.
point_on_ridge = np.array([1 / 2, 0, 0])
point_that_is_out = np.array([1 / 2, 1 / 3, 1 / 2])

# print(in_hull(vertices, point_that_is_in))
# print(in_hull(vertices, point_on_boundary))
# print(in_hull(vertices, point_on_ridge))
# print(in_hull(vertices, point_that_is_out))


# Also test with non-full-dimensional polytope
vertices = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
point_within = np.array([1 / 3, 1 / 3, 1 / 3])
point_within_approx = np.array([1 / 3 + 1e-8, 1 / 3, 1 / 3])
point_within_less_approx = np.array([1 / 3 + 1e-7, 1 / 3, 1 / 3])

# print(in_hull(vertices, point_within))
# print(in_hull(vertices, point_within_approx))
# print(in_hull(vertices, point_within_less_approx))


# Now test how scalable it is
n = 31744  # number of vertices
d = 64     # dimension of space
vertices = np.random.randint(0, 2, (n, d))  # random 0/1 polytope, i.e. random set of deterministic vertices
random_point = np.random.rand(d)
random_weight = np.random.rand(1)
point_within = random_weight * vertices[0] + (1-random_weight) * vertices[1]
now = time.time()
print(str(in_hull(vertices, random_point)) + ' and that took ' + str(time.time() - now) + ' seconds')
time.sleep(1)
now = time.time()
print(str(in_hull(vertices, point_within)) + ' and that took ' + str(time.time() - now) + ' seconds')
# The programme is correct all the time, but for point_within, it throws warnings that 'matrix is ill-conditioned' 
# and that result might be inaccurate (still returns True every time, as required).
# For random_point (which is almost always outside the convex hull) this warning is never thrown, so if a point is far 
# enough out there should probably be no problem.
"""


def list_of_fns_from_n_to_m(n, m):
    """ Returns list of all functions from [0,1,...,n] to [0,1,...,m]
    Each function is specified as a length-n array, so call f[k] to get f evaluated on k.
    Read how generators work at https://stackoverflow.com/questions/231767/what-does-the-yield-keyword-do """
    # We need a length-n array of numbers <m. Can use itertools.product to make n-fold cartesian product of m.
    return itertools.product(range(m), repeat=n)


def construct_NSS_to_full_matrix_but_weird(na, nb, nx, ny):
    """ Returns matrix of shape (nx*ny*na*nb, nx*ny*(na-1)(nb-1)+nx(na-1)+ny(nb-1)) that converts NSS coords
        {p(a|x), p(b|y), p(ab|xy)  |  a<na-1, b<nb-1, all x,y} =: NSS-I ∪ NSS-II ∪ NSS-III (see [p113])
    into 'weird' full coords
        {p(ab|xy)  |  (a,b) != (na-1,nb-1), all x,y} ∪ {p(na,nb|xy) - 1  |  all x,y}.
    NOTE This function assumes nx,ny,na,nb are all powers of 2, so that the coordinates can be labelled in the usual way.
    (specifically, this assumption is used in using concatenate_bits)
    """
    # Define dimensions
    full_dim = na * nb * nx * ny
    NSS_dim = dim_NSS(na, nb, nx, ny)
    matrix = np.zeros((full_dim, NSS_dim))

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
            matrix[current_row][get_NSS_III_index(a, b, x, y)] = 1
        elif a == na - 1 and b != nb - 1:
            # p(na-1 b|xy) = p(b|y) - sum_{_a < na-1} p(_a b|xy)
            matrix[current_row][get_NSS_II_index(b, y)] = 1
            for _a in range(0, na - 1):
                matrix[current_row][get_NSS_III_index(_a, b, x, y)] = -1
        elif a != na - 1 and b == nb - 1:
            # p(a nb-1|xy) = p(a|x) - sum_{_b < nb-1} p(a _b|xy)
            matrix[current_row][get_NSS_I_index(a, x)] = 1
            for _b in range(0, nb - 1):
                matrix[current_row][get_NSS_III_index(a, _b, x, y)] = -1
        elif a == na - 1 and b == nb - 1:
            # (For the current values of x,y), the rows corresponding to p(ab|xy) with (a,b)!=(na-1,nb-1) have already been filled.
            # So can use those to calculate the current row, which is -sum_{(a,b)!=(na-1,nb-1)} p(ab|xy)
            # NOTE that this isn't p(na-1,nb-1|xy) but instead it is p(na-1,nb-1|xy) - 1.
            for _a, _b in list(itertools.product(ra, rb))[0: na * nb - 1]:  # everything except (na-1, nb-1)
                matrix[current_row] -= matrix[_a * nb * nx * ny + _b * nx * ny + x * ny + y]
        else:
            print("something went wrong")
        current_row += 1
    return matrix


def construct_full_to_NSS_matrix(na, nb, nx, ny):
    """
    Make and return matrix that converts shape (na*nb*nx*ny) vectors (full dimension) to shape (dim_NSS(na, nb, nx, ny),)
    vectors (NSS representation). ith row of the matrix represents ith NSS coordinate
    """

    full_dim = na * nb * nx * ny
    NSS_dim = dim_NSS(na, nb, nx, ny)
    matrix = np.zeros((NSS_dim, full_dim))

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

    return matrix  # TODO test this function, and use it instead of special case in other file!


def dim_NSS(na, nb, nx, ny):
    return nx * ny * (na - 1) * (nb - 1) + nx * (na - 1) + ny * (nb - 1)
