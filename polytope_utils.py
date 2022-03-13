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
    A = np.r_[vertices.T, np.ones((1, n))]  # Transpose 'vertices', because in numpy they are provided as a column vector of row vectors
    b = np.r_[point, np.ones(1)]
    lp = linprog(np.zeros(n), A_eq=A, b_eq=b, options={'tol': tol})  # by default, bounds=(0,None), which enforces the constraint that the convex coefficients are non-negative.
    return lp.success  # solution of optimisation problem is not relevant - only if a solution was found


def in_affine_hull(points, point, tol=1e-8):
    """ Decide if point is in the convex hull of points.
    tol: tolerance value. Is passed on to scipy. Default value for scipy is 1e-8.
    Scipy linprog docs: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html"""
    n = len(points)  # number of vertices.
    A = np.r_[points.T, np.ones((1, n))]  # Transpose 'vertices', because in numpy they are provided as a column vector of row vectors
    b = np.r_[point, np.ones(1)]
    lp = linprog(np.zeros(n), A_eq=A, b_eq=b, options={'tol': tol}, bounds=(None, None))  # the only difference with convex hull: here we have no bounds on the coefficients
    return lp.success  # solution of optimisation problem is not relevant - only whether a solution was found


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
