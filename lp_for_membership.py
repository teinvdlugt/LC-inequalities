import sys

import numpy as np
from scipy.optimize import linprog

import symmetry_utils
import vector_space_utils as vs


def lp_without_vertices_nss_coords(p_test, tol=1e-8, method='interior-point', double_check_soln=False):
    """
    Uses the 'LP without vertices'/'LP with LC constraints' method to determine LC membership of p_test. See [p210].
    This method uses LP on 174 unknowns with 357 constraints and no objective function. TODO 377 can be reduced by using linear dependence of constraints
    scipy.optimize.linprog docs: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html
    :param p_test: Should be in NSS representation, so a length-86 vector. If it's length-87 then it's assumed to represent homogeneous coordinates.
    :param tol: tolerance value passed on to scipy.optimise.linprog. (irrelevant if using 'highs' method)
    :param method: method passed on to scipy.optimise.linprog. ‘highs-ds’, ‘highs-ipm’, ‘highs’, ‘interior-point’ (default), ‘revised simplex’, and ‘simplex’ (legacy) are supported.
    """
    assert len(p_test) in [86, 87]
    if len(p_test) == 87:
        p_test = 1. / p_test[-1] * np.array(p_test[:-1])

    # Constraint i: positivity of probabilities (see [p208,210] for matrix)
    cons_i_matrix = np.block([[vs.construct_NSS_to_full_homogeneous(), np.zeros((129, 87), dtype='int')],
                              [np.zeros((129, 87), dtype='int'), vs.construct_NSS_to_full_homogeneous()]])
    A_ub = -cons_i_matrix
    b_ub = np.zeros(258, dtype='int')
    assert A_ub.shape == (258, 174)

    # Constraint ii: p(λ=0) + p(λ=1) = 1
    cons_ii_matrix = np.zeros((1, 174), dtype='int')
    cons_ii_matrix[0, 86] = 1
    cons_ii_matrix[0, 173] = 1
    cons_ii_b = np.array([1, ])

    # Constraint iii: sum_λ p(...λ|..) = p_test(...|..)
    cons_iii_matrix = np.concatenate((np.identity(86), np.zeros((86, 1), dtype='int'),
                                      np.identity(86), np.zeros((86, 1), dtype='int')), axis=1)
    cons_iii_b = p_test

    # Constraint iv: iv-1: p(a1=0 b λ=0 | x1 0 y) - p(a1=0 b λ=0 | x1 1 y) = 0  for (b,y)≠(1,1)   (6 equalities)
    #                iv-2: p(a2=0 b λ=1 | 0 x2 y) - p(a2=0 b λ=1 | 1 x2 y) = 0  for (b,y)≠(1,1)   (6 equalities)
    cons_iv_matrix = np.zeros((12, 174), dtype='int')
    NtoF = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2)  # weirdness doesn't matter here
    # iv-1:  # NOTE this essentially constructs a (6,86) matrix which has as null space the 80-dim ahNSCO1 (subspace of ahNSS)
    for (b, y), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        current_row = vs.concatenate_bits(b, y, x1)
        # Find row vector that will give us p(a1=0 b λ=0 | x1 x2 y)
        # sum over a2, c
        for a2, c in vs.cart((0, 1), (0, 1)):
            cons_iv_matrix[current_row] += ((-1) ** x2) * np.r_[NtoF[vs.concatenate_bits(0, a2, c, b, x1, x2, y)], np.zeros(88, dtype='int')]
    # iv-2:
    for (b, y), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        current_row = 6 + vs.concatenate_bits(b, y, x2)
        # Find row vector that will give us p(a2=0 b λ=1 | x1 x2 y)
        # sum over a2, c
        for a1, c in vs.cart((0, 1), (0, 1)):
            cons_iv_matrix[current_row] += ((-1) ** x1) * np.r_[np.zeros(87, dtype='int'), NtoF[vs.concatenate_bits(a1, 0, c, b, x1, x2, y)], [0]]
    cons_iv_b = np.zeros(12, dtype='int')

    # The equality constraints ii-iv together:
    A_eq = np.r_[cons_ii_matrix, cons_iii_matrix, cons_iv_matrix]
    b_eq = np.r_[cons_ii_b, cons_iii_b, cons_iv_b]
    assert A_eq.shape == (99, 174) and b_eq.shape == (99,)

    # Do LP
    lp = linprog(np.zeros(174, dtype='int'), A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options={'tol': tol}, method=method)

    # Double-check that solution is correct (if solution found)
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:87] is in NSCO1
        p1_nss_homog = lp.x[:87]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2y,λ=0)
        p1_full_homog = vs.construct_NSS_to_full_homogeneous() @ p1_nss_homog
        assert vs.is_in_NSCO1(p1_full_homog, tol)

        # check that p2 := lp.x[87:174] is in NSCO2
        p2_nss_homog = lp.x[87:174]
        p2_full_homog = vs.construct_NSS_to_full_homogeneous() @ p2_nss_homog
        swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
        p2_full_homog_swapped = swap_A1_A2_matrix @ p2_full_homog
        assert vs.is_in_NSCO1(p2_full_homog_swapped, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_test
        sum_p1_p2 = p1_nss_homog[:-1] + p2_nss_homog[:-1]
        assert np.all(np.abs(sum_p1_p2 - p_test) < tol)

    return lp


def in_convex_hull_lp(vertices, point, tol=1e-8):
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


def in_affine_hull_lp(points, point, tol=1e-8):
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

# print(in_convex_hull_lp(vertices, point_that_is_in))
# print(in_convex_hull_lp(vertices, point_on_boundary))
# print(in_convex_hull_lp(vertices, point_on_ridge))
# print(in_convex_hull_lp(vertices, point_that_is_out))


# Also test with non-full-dimensional polytope
vertices = np.array([[1, 0, 0],
                     [0, 1, 0],
                     [0, 0, 1]])
point_within = np.array([1 / 3, 1 / 3, 1 / 3])
point_within_approx = np.array([1 / 3 + 1e-8, 1 / 3, 1 / 3])
point_within_less_approx = np.array([1 / 3 + 1e-7, 1 / 3, 1 / 3])

# print(in_convex_hull_lp(vertices, point_within))
# print(in_convex_hull_lp(vertices, point_within_approx))
# print(in_convex_hull_lp(vertices, point_within_less_approx))


# Now test how scalable it is
n = 31744  # number of vertices
d = 64     # dimension of space
vertices = np.random.randint(0, 2, (n, d))  # random 0/1 polytope, i.e. random set of deterministic vertices
random_point = np.random.rand(d)
random_weight = np.random.rand(1)
point_within = random_weight * vertices[0] + (1-random_weight) * vertices[1]
now = time.time()
print(str(in_convex_hull_lp(vertices, random_point)) + ' and that took ' + str(time.time() - now) + ' seconds')
time.sleep(1)
now = time.time()
print(str(in_convex_hull_lp(vertices, point_within)) + ' and that took ' + str(time.time() - now) + ' seconds')
# The programme is correct all the time, but for point_within, it throws warnings that 'matrix is ill-conditioned' 
# and that result might be inaccurate (still returns True every time, as required).
# For random_point (which is almost always outside the convex hull) this warning is never thrown, so if a point is far 
# enough out there should probably be no problem.
"""


def test_membership_of_quantum_cors(lp_method=lp_without_vertices_nss_coords, quantum_cor_file='panda-files/some_quantum_cors3.npy', tol=1e-12, method='highs', double_check_soln=False):
    qm_cors = np.load(quantum_cor_file).astype('float64')
    cors_not_in_LC = []
    cors_not_in_LC_indices = []
    for i in range(0, len(qm_cors)):
        lp_result = lp_method(qm_cors[i], tol, method, double_check_soln=double_check_soln)
        if not lp_result.success:
            # print('Found a correlation that is not in LC!')
            cors_not_in_LC.append(qm_cors[i])
            cors_not_in_LC_indices.append(i)
            # return lp_result, i, qm_cors[i]
        print('checked %d / %d correlations, %d of which not in LC' % (i + 1, len(qm_cors), len(cors_not_in_LC)), end='\r')
        sys.stdout.flush()
    print('Finished checking all cors')
    return cors_not_in_LC, cors_not_in_LC_indices


if __name__ == '__main__':
    cors, cors_indices = test_membership_of_quantum_cors(quantum_cor_file='panda-files/some_quantum_cors5.npy', double_check_soln=True, tol=1e-6)
    # np.save('cors_not_in_LC', cors)

    """
    tols = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]
    thresholds = []
    for cor in np.load('cors_not_in_LC.npy'):
        for t in range(0, len(tols)):
            if not lp_without_vertices_nss_coords(cor, tols[t]).success:
                print('Threshold 1e-%d' % (t+1))
                thresholds.append('1e-%d' % (t+1))
                break
    """
