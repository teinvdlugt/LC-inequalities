import itertools
import sys

import numpy as np
from scipy.optimize import linprog

import quantum_utils as qm
import symmetry_utils
import towards_lc
import utils
import vector_space_utils as vs

dim_nss = vs.dim_NSS(8, 2, 4, 2)


def lp_constraints_is_valid_lc_deco():
    """ Returns arrays A_ub, b_ub, A_eq, B_eq that represent the constraint that a (86+1)*2=174-length vector represents
         a valid 'LC decomposition', i.e. of the form λp^1 + (1-λ)p^2. """
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

    # Constraint iii: iii-1: p(a1=0 b λ=0 | x1 0 y) - p(a1=0 b λ=0 | x1 1 y) = 0  for (b,y)≠(1,1)   (6 equalities)
    #                iii-2: p(a2=0 b λ=1 | 0 x2 y) - p(a2=0 b λ=1 | 1 x2 y) = 0  for (b,y)≠(1,1)   (6 equalities)
    cons_iii_matrix = np.zeros((12, 174), dtype='int')
    NtoF = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2)  # weirdness doesn't matter here
    # iii-1:  # NOTE this essentially constructs a (6,86) matrix which has as null space the 80-dim ahNSCO1 (subspace of ahNSS)
    for (b, y), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        current_row = vs.concatenate_bits(b, y, x1)
        # Find row vector that will give us p(a1=0 b λ=0 | x1 x2 y)
        # sum over a2, c
        for a2, c in vs.cart((0, 1), (0, 1)):
            cons_iii_matrix[current_row] += ((-1) ** x2) * np.r_[NtoF[vs.concatenate_bits(0, a2, c, b, x1, x2, y)], np.zeros(88, dtype='int')]
    # iii-2:
    for (b, y), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        current_row = 6 + vs.concatenate_bits(b, y, x2)
        # Find row vector that will give us p(a2=0 b λ=1 | x1 x2 y)
        # sum over a2, c
        for a1, c in vs.cart((0, 1), (0, 1)):
            cons_iii_matrix[current_row] += ((-1) ** x1) * np.r_[np.zeros(87, dtype='int'), NtoF[vs.concatenate_bits(a1, 0, c, b, x1, x2, y)], [0]]
    cons_iii_b = np.zeros(12, dtype='int')

    # The equality constraints ii and iii together:
    A_eq = np.r_[cons_ii_matrix, cons_iii_matrix]
    b_eq = np.r_[cons_ii_b, cons_iii_b]

    return A_ub, b_ub, A_eq, b_eq


def lp_is_cor_in_lc(p_test, tol=1e-12, method='highs', double_check_soln=False):
    """
    Uses the 'LP without vertices'/'LP with LC constraints' method to determine LC membership of p_nss. See [p210].
    This method uses LP on 174 unknowns with 357 constraints and no objective function. TODO 377 can be reduced by using linear dependence of constraints
    scipy.optimize.linprog docs: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html
    :param p_test: Should be in NSS representation, so a length-86 vector. If it's length-87 then it's assumed to represent homogeneous coordinates.
    :param tol: tolerance value passed on to scipy.optimise.linprog. (irrelevant if using 'highs' method)
    :param method: method passed on to scipy.optimise.linprog. ‘highs-ds’, ‘highs-ipm’, ‘highs’, ‘interior-point’ (default), ‘revised simplex’, and ‘simplex’ (legacy) are supported.
    """
    assert len(p_test) in [86, 87]
    if len(p_test) == 87:
        p_test = 1. / p_test[-1] * np.array(p_test[:-1])

    A_ub, b_ub, A_eq, b_eq = lp_constraints_is_valid_lc_deco()

    # Remaining constraint: sum_λ p(...λ|..) = p_nss(...|..)
    p_nss_cons_matrix = np.concatenate((np.identity(86), np.zeros((86, 1), dtype='int'),
                                        np.identity(86), np.zeros((86, 1), dtype='int')), axis=1)
    p_nss_cons_b = p_test

    A_eq = np.r_[A_eq, p_nss_cons_matrix]
    b_eq = np.r_[b_eq, p_nss_cons_b]

    # Do LP
    lp = linprog(np.zeros(174, dtype='int'), A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options={'tol': tol}, method=method)

    # Double-check that solution is correct (if solution found)
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:87] is in NSCO1
        p1_nss_homog = lp.x[:87]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2y,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_homogeneous() @ p1_nss_homog
            assert vs.is_in_NSCO1(p1_full_homog, tol)

        # check that p2 := lp.x[87:174] is in NSCO2
        p2_nss_homog = lp.x[87:174]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_homogeneous() @ p2_nss_homog
            swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm_homog(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
            p2_full_homog_swapped = swap_A1_A2_matrix @ p2_full_homog
            assert vs.is_in_NSCO1(p2_full_homog_swapped, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_nss
        sum_p1_p2 = p1_nss_homog[:-1] + p2_nss_homog[:-1]
        assert np.all(np.abs(sum_p1_p2 - p_test) < tol)

    return lp


def lp_max_violation_by_LC(ineq, method='highs', tol=1e-12, double_check_soln=True):
    ineq = np.array(ineq)
    assert ineq.shape == (dim_nss + 1,)

    A_ub, b_ub, A_eq, b_eq = lp_constraints_is_valid_lc_deco()

    # Objective function to minimise: -ineq @ (λp^1 + (1-λ)p^2)
    matrix = np.concatenate((np.identity(dim_nss + 1), np.identity(dim_nss + 1)), axis=1)  # multiplying the vector of unknowns with this gives the homogeneous nss representation of (λp^1 + (1-λ)p^2)
    c = -ineq @ matrix

    # Do LP
    options = None if method == 'highs' else {'tol': tol}
    lp = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options=options, method=method)

    assert lp.success

    # Check solution
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:87] is in NSCO1
        p1_nss_homog = lp.x[:87]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2y,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_homogeneous() @ p1_nss_homog
            assert vs.is_in_NSCO1(p1_full_homog, tol)

        # check that p2 := lp.x[87:174] is in NSCO2
        p2_nss_homog = lp.x[87:174]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_homogeneous() @ p2_nss_homog
            swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm_homog(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
            p2_full_homog_swapped = swap_A1_A2_matrix @ p2_full_homog
            assert vs.is_in_NSCO1(p2_full_homog_swapped, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_nss
        sum_p1_p2_h = p1_nss_homog + p2_nss_homog
        assert abs(lp.fun + ineq @ sum_p1_p2_h) < tol

    return -lp.fun


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


def test_membership_of_quantum_cors(lp_method=lp_is_cor_in_lc, quantum_cor_file='panda-files/some_quantum_cors3.npy', tol=1e-12, method='highs', double_check_soln=False):
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


def construct_ineq_full(num_of_binary_vars, point_function, upper_bound=0.0):
    """ A function of n binary variables, where n is often equal to 7, which returns the points won by the parties when they output those binary variables. """
    ineq_full = np.zeros((2,) * num_of_binary_vars)
    for var_tuple in itertools.product((0, 1), repeat=num_of_binary_vars):
        ineq_full[var_tuple] += point_function(*var_tuple)
    ineq_full_h = np.r_[ineq_full.flatten(), [-upper_bound]]
    return ineq_full_h


def construct_ineq_nss(num_of_binary_vars, point_function, upper_bound=0.0, na=8, nb=2, nx=4, ny=2):
    """ A function of n binary variables, where n is often equal to 7, which returns the points won by the parties when they output those binary variables. """
    ineq_nss_h = vs.construct_NSS_to_full_homogeneous(na, nb, nx, ny).T @ construct_ineq_full(num_of_binary_vars, point_function, upper_bound)
    return ineq_nss_h


def print_x12y_chsh_violations(p_acbxy_nss):
    p_full_h = vs.construct_NSS_to_full_homogeneous(8, 2, 4, 2) @ p_acbxy_nss
    p_full = (1 / p_full_h[-1] * p_full_h[:-1]).reshape((2,) * 7)
    bell_facets = 1 / 2 * utils.read_vertex_range_from_file('panda-files/chsh_2222')
    p_cbx1x2y = np.einsum('aecbxwy->cbxwy', p_full)
    for x2 in (0, 1):
        cor = np.r_[vs.construct_full_to_NSS_matrix(2, 2, 2, 2) @ p_cbx1x2y[:, :, :, x2, :].reshape(16), [1]]
        print('CHSH violation for x2=%d: %s' % (x2, str(utils.max_violation_h(cor, bell_facets))))
    for x1 in (0, 1):
        cor = np.r_[vs.construct_full_to_NSS_matrix(2, 2, 2, 2) @ p_cbx1x2y[:, :, x1, :, :].reshape(16), [1]]
        print('CHSH violation for x1=%d: %s' % (x1, str(utils.max_violation_h(cor, bell_facets))))


def max_of_all_chsh_violations(p_acbxy_nss):
    p_full_h = vs.construct_NSS_to_full_homogeneous(8, 2, 4, 2) @ p_acbxy_nss
    p_full = (1 / p_full_h[-1] * p_full_h[:-1]).reshape((2,) * 7)
    bell_facets = 1 / 2 * utils.read_vertex_range_from_file('panda-files/chsh_2222')
    # CHSH between c and b:
    p_cbx1x2y = np.einsum('aecbxwy->cbxwy', p_full)
    cor1 = p_cbx1x2y[:, :, :, 0, :].flatten()
    cor2 = p_cbx1x2y[:, :, :, 1, :].flatten()
    cor3 = p_cbx1x2y[:, :, 0, :, :].flatten()
    cor4 = p_cbx1x2y[:, :, 1, :, :].flatten()
    # CHSH between a1 and b:
    p_a1bx1x2y = np.einsum('aecbxwy->abxwy', p_full)
    cor5 = p_a1bx1x2y[:, :, :, 0, :].flatten()
    cor6 = p_a1bx1x2y[:, :, :, 1, :].flatten()
    cor7 = p_a1bx1x2y[:, :, 0, :, :].flatten()
    cor8 = p_a1bx1x2y[:, :, 1, :, :].flatten()
    # CHSH between a2 and b:
    p_a2bx1x2y = np.einsum('aecbxwy->ebxwy', p_full)
    cor9 = p_a2bx1x2y[:, :, :, 0, :].flatten()
    cor10 = p_a2bx1x2y[:, :, :, 1, :].flatten()
    cor11 = p_a2bx1x2y[:, :, 0, :, :].flatten()
    cor12 = p_a2bx1x2y[:, :, 1, :, :].flatten()
    all_cors_full = np.array([cor1, cor2, cor3, cor4, cor5, cor6, cor7, cor8, cor9, cor10, cor11, cor12])
    all_cors_nss_h = (np.concatenate((all_cors_full, np.ones((len(all_cors_full), 1), dtype='int')), axis=1)) @ vs.construct_full_to_NSS_homog(2, 2, 2, 2).T

    return utils.max_violation_h(all_cors_nss_h, bell_facets)


if __name__ == '__main__':
    # print(lp_is_cor_in_lc(
    #     qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_ctb_plusphiplus,
    #                               instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
    #                               instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
    #                               instr_C=qm.instr_vn_destr(qm.x_onb),
    #                               instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])
    # ).success)
    # print(lp_is_cor_in_lc(
    #     qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_ctb_plusphiplus,
    #                               instrs_A1=[qm.instr_measure_and_prepare(qm.diag1_onb, qm.diag1_onb[0]), qm.instr_measure_and_prepare(qm.diag2_onb, qm.diag1_onb[1])],
    #                               instrs_A2=[qm.instr_measure_and_prepare(qm.diag1_onb, qm.diag1_onb[0]), qm.instr_measure_and_prepare(qm.diag2_onb, qm.diag1_onb[1])],
    #                               instr_C=qm.instr_vn_destr(qm.x_onb),
    #                               instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.x_onb)])
    # ).success)
    # print(lp_is_cor_in_lc(
    #     qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_ctb_plusphiplus,
    #                               instrs_A1=[qm.instr_measure_and_prepare(qm.diag1_onb, qm.diag1_onb[0]), qm.instr_measure_and_prepare(qm.diag2_onb, qm.diag1_onb[1])],
    #                               instrs_A2=[qm.instr_measure_and_prepare(qm.diag1_onb, qm.diag1_onb[0]), qm.instr_measure_and_prepare(qm.diag2_onb, qm.diag1_onb[1])],
    #                               instr_C=qm.instr_vn_destr(qm.x_onb),
    #                               instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])
    # ).success)
    """cor = qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_tcb_0phi,  # NOTE qm.rho_ctb_plusphiplus also works, if Alice does X mmt on setting x_i=1. Think about later
                                    instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                    instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                    instr_C=qm.instr_vn_destr(qm.onb_from_direction(0.148 * np.pi)),  # 0.148 * np.pi seems to give maximal violation
                                    instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.x_onb)])
    print(lp_is_cor_in_lc(cor).success)

    print('-- CHSH stuff:')
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: ((b + c) % 2 == x1 * y and x1 == 0 and y == 0 and x2 == 0) * 1) @ cor)
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: ((b + c) % 2 == x1 * y and x1 == 0 and y == 1 and x2 == 0) * 1) @ cor)
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: ((b + c) % 2 == x1 * y and x1 == 1 and y == 0 and x2 == 0) * 1) @ cor)
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: ((b + c) % 2 == x1 * y and x1 == 1 and y == 1 and x2 == 0) * 1) @ cor)
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4) @ cor)
    print('-- α stuff:')
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2) @ cor)
    print(construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2) @ cor)
    print('-- Total:')"""

    x1y_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0) * 1 / 4 +
                                                                     (b == 1 and a1 == x2 and y == 0) * 1 / 4 +
                                                                     ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)
    x1y_different_y_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0) * 1 / 4 +
                                                                                 (b == 1 and a1 == x2 and y == 0) * 1 / 4 +
                                                                                 ((b + c) % 2 == x1 * (1 - y) and x2 == 0) * 1 / 4, 7 / 4)
    x1y_lazy_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2 +
                                                                          (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2 +
                                                                          ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)
    x1y_different_y_lazy_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2 +
                                                                                      (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2 +
                                                                                      ((b + c) % 2 == x1 * (1 - y) and x2 == 0) * 1 / 4, 7 / 4)
    x1y_unlazy_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0) * 1 / 4 +
                                                                            (b == 1 and a1 == x2 and y == 0) * 1 / 4 +
                                                                            ((b + c) % 2 == x1 * y) * 1 / 8, 7 / 4)
    x1y_extra_unlazy_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1) * 1 / 8 +
                                                                                  (b == 1 and a1 == x2) * 1 / 8 +
                                                                                  ((b + c) % 2 == x1 * y) * 1 / 8, 7 / 4)
    x1y_lazy_different_x2_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2 +
                                                                                       (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2 +
                                                                                       ((b + c) % 2 == x1 * y and x2 == 1) * 1 / 4, 7 / 4)
    x1y_lazy_different_x2y_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 1 and x2 == 0) * 1 / 2 +
                                                                                        (b == 1 and a1 == x2 and y == 1 and x1 == 0) * 1 / 2 +
                                                                                        ((b + c) % 2 == x1 * y and x2 == 1) * 1 / 4, 7 / 4)
    x1y_lazy_different_x1x2_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2 +
                                                                                         (b == 1 and a1 == x2 and y == 0 and x1 == 1) * 1 / 2 +
                                                                                         ((b + c) % 2 == x1 * y and x2 == 1) * 1 / 4, 7 / 4)
    x1y_lazy_different_x1x2y_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 1 and x2 == 0) * 1 / 2 +
                                                                                          (b == 1 and a1 == x2 and y == 1 and x1 == 1) * 1 / 2 +
                                                                                          ((b + c) % 2 == x1 * y and x2 == 1) * 1 / 4, 7 / 4)
    x1y_lazy_different_x1_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2 +
                                                                                       (b == 1 and a1 == x2 and y == 0 and x1 == 1) * 1 / 2 +
                                                                                       ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)
    x1y_lazy_different_x1y_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 1 and x2 == 0) * 1 / 2 +
                                                                                        (b == 1 and a1 == x2 and y == 1 and x1 == 1) * 1 / 2 +
                                                                                        ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)
    # Violated faces and facets:
    x1y_lazy_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 0) * 1 / 2 +
                                                                          (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2 +
                                                                          ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)  # dim 73
    x1y_varx2_lazy_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 1) * 1 / 2 +
                                                                                (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2 +
                                                                                ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)
    x1y_varx2_lazy_plus_last_term_ineq = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 1) * 1 / 2 +
                                                                                               (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2 +
                                                                                               ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4 +
                                                                                               (a1 == 1 and c == 1 - b == y and x1 == 0 and x2 == 0) * 1 / 2, 7 / 4)
    just_the_alpha_terms = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == 0 and a2 == x1 and y == 0 and x2 == 1) * 1 / 2 +
                                                                                 (b == 1 and a1 == x2 and y == 0 and x1 == 0) * 1 / 2, 1)
    just_the_chsh_term = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 3 / 4)
    just_the_last_term = construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (a1 == 1 and c == 1 - b == y and x1 == 0 and x2 == 0) * 1 / 2)


    # Correlations that still require an explanation:
    """
    cor = qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_tcb_0phi,
                                    instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                    instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                    instr_C=qm.instr_vn_destr(qm.diag1_onb),
                                    instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.diag1_onb)])
    # or:                           instr_C=qm.instr_vn_destr(qm.diag1_onb),
    #                               instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])
    print(lp_is_cor_in_lc(cor).success)  # is outside LC
    print_x12y_chsh_violations(cor)  # but doesn't violate a CHSH ineq btw b and c (neither for fixed x1 nor for fixed x2)
    print(max_of_all_chsh_violations(cor))  # and also not between b and a1 or between b and a2.
    """

    """cor = qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_ctb_plusphiplus,
                                    instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
                                    instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
                                    instr_C=qm.instr_vn_destr(qm.x_onb),
                                    instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])
    print(lp_is_cor_in_lc(cor).success)  # is outside LC
    print_x12y_chsh_violations(cor)  # but doesn't violate a CHSH ineq btw b and c (neither for fixed x1 nor for fixed x2)
    print(max_of_all_chsh_violations(cor))"""

    print('Violation of alpha term:', just_the_alpha_terms @ cor)
    print('Violation of CHSH term:', just_the_chsh_term @ cor)
    print('Value of last term:', just_the_last_term @ cor)
    print('Violation of the entire expression:', x1y_varx2_lazy_plus_last_term_ineq @ cor)
