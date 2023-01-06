import itertools
import sys

import numpy as np
import scipy.optimize

import utils
import vector_space_utils as vs


## TESTING MEMBERSHIP OF POLYTOPE FOR CORRELATIONS
def lp_constraints_is_valid_lc_deco():
    """ Returns arrays A_ub, b_ub, A_eq, B_eq that represent the constraint that a (86+1)*2=174-length vector represents
         a valid 'LC decomposition', i.e. of the form λp^1 + (1-λ)p^2. """
    # Constraint i: positivity of probabilities (see [p208,210] for matrix)
    cons_i_matrix = np.block([[vs.construct_NSS_to_full_h(), np.zeros((129, 87), dtype='int')],
                              [np.zeros((129, 87), dtype='int'), vs.construct_NSS_to_full_h()]])
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
    NtoF = vs.construct_NSS_to_full_h(8, 2, 4, 2)
    # iii-1:  # NOTE this essentially constructs a (6,86) matrix which has as null space the 80-dim ahNSCO1 (subspace of ahNSS)
    for (b, y), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        current_row = utils.concatenate_bits(b, y, x1)
        # Find row vector that will give us p(a1=0 b λ=0 | x1 x2 y)
        # sum over a2, c
        for a2, c in vs.cart((0, 1), (0, 1)):
            cons_iii_matrix[current_row] += ((-1) ** x2) * np.r_[NtoF[utils.concatenate_bits(0, a2, c, b, x1, x2, y)], np.zeros(87, dtype='int')]
    # iii-2:
    for (b, y), x1, x2 in vs.cart(vs.cart((0, 1), (0, 1))[:-1], (0, 1), (0, 1)):
        current_row = 6 + utils.concatenate_bits(b, y, x2)
        # Find row vector that will give us p(a2=0 b λ=1 | x1 x2 y)
        # sum over a2, c
        for a1, c in vs.cart((0, 1), (0, 1)):
            cons_iii_matrix[current_row] += ((-1) ** x1) * np.r_[np.zeros(87, dtype='int'), NtoF[utils.concatenate_bits(a1, 0, c, b, x1, x2, y)]]
    cons_iii_b = np.zeros(12, dtype='int')

    # The equality constraints ii and iii together:
    A_eq = np.r_[cons_ii_matrix, cons_iii_matrix]
    b_eq = np.r_[cons_ii_b, cons_iii_b]

    return A_ub, b_ub, A_eq, b_eq


def is_cor_in_lc(p_test, tol=1e-12, method='highs', double_check_soln=False):
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
    options = None if method == 'highs' else {'tol': tol}
    lp = scipy.optimize.linprog(np.zeros(174, dtype='int'), A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options=options, method=method)

    # Double-check that solution is correct (if solution found)
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:87] is in NSCO1
        p1_nss_homog = lp.x[:87]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2y,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_h() @ p1_nss_homog
            assert vs.is_in_LC1(p1_full_homog, tol)

        # check that p2 := lp.x[87:174] is in NSCO2
        p2_nss_homog = lp.x[87:174]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_h() @ p2_nss_homog
            assert vs.is_in_LC2(p2_full_homog, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_nss
        sum_p1_p2 = p1_nss_homog[:-1] + p2_nss_homog[:-1]
        assert np.all(np.abs(sum_p1_p2 - p_test) < tol)

    return lp


def test_membership_of_quantum_cors(lp_method=is_cor_in_lc, quantum_cor_file='panda-files/some_quantum_cors3.npy', tol=1e-12, method='highs', double_check_soln=False):
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


def lp_constraints_is_valid_LCstar_deco():
    # Constraint i: positivity of probabilities (see [p208,210] for matrix)
    cons_i_matrix = np.block([[vs.construct_NSS_to_full_h(), np.zeros((129, 87), dtype='int')],
                              [np.zeros((129, 87), dtype='int'), vs.construct_NSS_to_full_h()]])
    A_ub = -cons_i_matrix
    b_ub = np.zeros(258, dtype='int')
    assert A_ub.shape == (258, 174)

    # Constraint ii: p(λ=0) + p(λ=1) = 1
    cons_ii_matrix = np.zeros((1, 174), dtype='int')
    cons_ii_matrix[0, 86] = 1
    cons_ii_matrix[0, 173] = 1
    cons_ii_b = np.array([1, ])

    # Constraint iii: iii-1: p(a1=0 λ=0 | x1 0 0) - p(a1=0 λ=0 | x1 1 0) = 0  for x1=0,1  (2 equalities)
    #                 iii-2: p(a2=0 λ=1 | 0 x2 0) - p(a2=0 λ=1 | 1 x2 0) = 0  for x2=0,1  (2 equalities)
    cons_iii_matrix = np.zeros((4, 174), dtype='int')
    NtoF = vs.construct_NSS_to_full_h(8, 2, 4, 2)
    # iii-1:  # NOTE this essentially constructs a (6,86) matrix which has as null space the 80-dim ahNSCO1 (subspace of ahNSS)
    for x1, x2 in vs.cart((0, 1), (0, 1)):
        current_row = x1
        for a2, c, b in vs.cart((0, 1), (0, 1), (0, 1)):
            cons_iii_matrix[current_row] += ((-1) ** x2) * np.r_[NtoF[utils.concatenate_bits(0, a2, c, b, x1, x2, 0)], np.zeros(87, dtype='int')]
    # iii-2:
    for x1, x2 in vs.cart((0, 1), (0, 1)):
        current_row = 2 + x2
        # Find row vector that will give us p(a2=0 b λ=1 | x1 x2 y)
        # sum over a2, c
        for a1, c, b in vs.cart((0, 1), (0, 1), (0, 1)):
            cons_iii_matrix[current_row] += ((-1) ** x1) * np.r_[np.zeros(87, dtype='int'), NtoF[utils.concatenate_bits(a1, 0, c, b, x1, x2, 0)]]
    cons_iii_b = np.zeros(4, dtype='int')

    # The equality constraints ii and iii together:
    A_eq = np.r_[cons_ii_matrix, cons_iii_matrix]
    b_eq = np.r_[cons_ii_b, cons_iii_b]

    return A_ub, b_ub, A_eq, b_eq


def is_cor_in_LCstar(p_test, tol=1e-12, method='highs', double_check_soln=False):
    assert len(p_test) in [86, 87]
    if len(p_test) == 87:
        p_test = 1. / p_test[-1] * np.array(p_test[:-1])

    A_ub, b_ub, A_eq, b_eq = lp_constraints_is_valid_LCstar_deco()

    # Remaining constraint: sum_λ p(...λ|..) = p_nss(...|..)
    p_nss_cons_matrix = np.concatenate((np.identity(86), np.zeros((86, 1), dtype='int'),
                                        np.identity(86), np.zeros((86, 1), dtype='int')), axis=1)
    p_nss_cons_b = p_test

    A_eq = np.r_[A_eq, p_nss_cons_matrix]
    b_eq = np.r_[b_eq, p_nss_cons_b]

    # Do LP
    lp = scipy.optimize.linprog(np.zeros(174, dtype='int'), A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options={'tol': tol}, method=method)

    # Double-check that solution is correct (if solution found)
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:87] is in NSCO1
        p1_nss_homog = lp.x[:87]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2y,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_h() @ p1_nss_homog
            assert vs.is_in_LC1st(p1_full_homog, tol)

        # check that p2 := lp.x[87:174] is in NSCO2
        p2_nss_homog = lp.x[87:174]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_h() @ p2_nss_homog
            assert vs.is_in_LC2st(p2_full_homog, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_nss
        sum_p1_p2 = p1_nss_homog[:-1] + p2_nss_homog[:-1]
        assert np.all(np.abs(sum_p1_p2 - p_test) < tol)

    return lp


## COMPUTING MAXIMAL VIOLATIONS OF AN INEQUALITY BY THE POLYTOPE
def max_violation_by_LC(ineq, method='highs', tol=1e-12, double_check_soln=True):
    ineq = np.array(ineq)
    dim_nss = vs.dim_NSS(8, 2, 4, 2)
    assert ineq.shape == (dim_nss + 1,)

    A_ub, b_ub, A_eq, b_eq = lp_constraints_is_valid_lc_deco()

    # Objective function to minimise: -ineq @ (λp^1 + (1-λ)p^2)
    matrix = np.concatenate((np.identity(dim_nss + 1), np.identity(dim_nss + 1)), axis=1)  # multiplying the vector of unknowns with this gives the homogeneous nss representation of (λp^1 + (1-λ)p^2)
    c = -ineq @ matrix

    # Do LP
    options = None if method == 'highs' else {'tol': tol}
    lp = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options=options, method=method)

    assert lp.success

    # Check solution
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:87] is in LC1
        p1_nss_homog = lp.x[:87]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2y,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_h() @ p1_nss_homog
            assert vs.is_in_LC1(p1_full_homog, tol)

        # check that p2 := lp.x[87:174] is in LC2
        p2_nss_homog = lp.x[87:174]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_h() @ p2_nss_homog
            assert vs.is_in_LC2(p2_full_homog, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_nss
        sum_p1_p2_h = p1_nss_homog + p2_nss_homog
        assert abs(lp.fun + ineq @ sum_p1_p2_h) < tol

    return -lp.fun, lp.x


def lp_constraints_is_valid_LC1():
    """ Returns arrays A_ub, b_ub, A_eq, B_eq that represent the constraint that an 87-length vector is in LC1*. """
    NtoFh = vs.construct_NSS_to_full_h()
    # Constraint i: positivity of probabilities (see [p208,210] for matrix)
    cons_i_matrix = NtoFh
    A_ub = -NtoFh[:-1]  # all rows except for the last (the homogeneous scaling)
    b_ub = np.zeros(128, dtype='int')
    assert A_ub.shape == (128, 87)

    # Constraint ii: p(a1=0 b | x1 0 y) - p(a1=0 | x1 1 y) = 0  for b,x1,y=0,1   (8 equalities)
    A_eq = np.zeros((8, 87), dtype='int')
    for a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        current_row = utils.concatenate_bits(b, x1, y)
        A_eq[current_row] += ((-1) ** x2) * NtoFh[utils.concatenate_bits(0, a2, c, b, x1, x2, y)]
    b_eq = np.zeros(8, dtype='int')

    # Constraint iii: the last coordinate (the homogeneous scaling) is 1.
    A_eq = np.r_[A_eq, [vs.one_hot_vector(87, -1)]]
    b_eq = np.r_[b_eq, [1]]

    # N.B. NSS constraint is already covered by using NSS coordinates

    return A_ub, b_ub, A_eq, b_eq


def lp_constraints_is_valid_LC1star():
    """ Returns arrays A_ub, b_ub, A_eq, B_eq that represent the constraint that an 87-length vector is in LC1*. """
    NtoFh = vs.construct_NSS_to_full_h()
    # Constraint i: positivity of probabilities (see [p208,210] for matrix)
    cons_i_matrix = NtoFh
    A_ub = -NtoFh[:-1]  # all rows except for the last (the homogeneous scaling)
    b_ub = np.zeros(128, dtype='int')
    assert A_ub.shape == (128, 87)

    # Constraint ii: p(a1=0 | x1 0 0) - p(a1=0 | x1 1 0) = 0  for x1=0,1   (2 equalities)
    A_eq = np.zeros((2, 87), dtype='int')
    for a2, c, b, x1, x2 in itertools.product((0, 1), repeat=5):
        current_row = x1
        A_eq[current_row] += ((-1) ** x2) * NtoFh[utils.concatenate_bits(0, a2, c, b, x1, x2, 0)]
    b_eq = np.zeros(2, dtype='int')

    # Constraint iii: the last coordinate (the homogeneous scaling) is 1.
    A_eq = np.r_[A_eq, [vs.one_hot_vector(87, -1)]]
    b_eq = np.r_[b_eq, [1]]

    # N.B. NSS constraint is already covered by using NSS coordinates

    return A_ub, b_ub, A_eq, b_eq


def max_violation_under_constraints(ineq, constraints, method='highs', tol=1e-12):
    ineq = np.array(ineq)
    dim_nss = vs.dim_NSS(8, 2, 4, 2)
    assert ineq.shape == (dim_nss + 1,)

    A_ub, b_ub, A_eq, b_eq = constraints  # e.g. lp_constraints_is_valid_LC1() or lp_constraints_is_valid_LC1star()

    # Objective function to minimise: -ineq @ cor
    c = -ineq

    # Do LP
    options = None if method == 'highs' else {'tol': tol}
    lp = scipy.optimize.linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options=options, method=method)

    assert lp.success
    return -lp.fun


def max_violation_by_LC1(ineq, method='highs', tol=1e-12):
    return max_violation_under_constraints(ineq, lp_constraints_is_valid_LC1(), method=method, tol=tol)


def max_violation_by_LC1st(ineq, method='highs', tol=1e-12):
    return max_violation_under_constraints(ineq, lp_constraints_is_valid_LC1star(), method=method, tol=tol)
