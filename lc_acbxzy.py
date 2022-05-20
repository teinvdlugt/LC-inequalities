import functools
import itertools
import math

import numpy as np
from scipy.optimize import linprog

import symmetry_utils
import towards_lc
import utils
import vector_space_utils as vs
import quantum_utils as qm

dim_nss = vs.dim_NSS(8, 2, 8, 2)
dim_full = 2 ** 8
B = (0, 1)


def constraints_is_valid_lc_deco():
    """ Returns arrays A_ub, b_ub, A_eq, B_eq that represent the constraint that a (170+1)*2=342-length vector represents
     a valid 'LC decomposition', i.e. of the form λp^1 + (1-λ)p^2. """
    num_of_unknowns = (dim_nss + 1) * 2

    # Constraint i: positivity of probabilities (see [p208,210] for matrix)
    cons_i_matrix = np.block([[vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2), np.zeros((dim_full + 1, dim_nss + 1), dtype='int')],
                              [np.zeros((dim_full + 1, dim_nss + 1), dtype='int'), vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2)]])
    A_ub = -cons_i_matrix
    b_ub = np.zeros(2 * dim_full + 2, dtype='int')
    assert A_ub.shape == (2 * dim_full + 2, num_of_unknowns)

    # Constraint ii: p(λ=0) + p(λ=1) = 1
    cons_ii_matrix = np.zeros((1, num_of_unknowns), dtype='int')
    cons_ii_matrix[0, dim_nss] = 1
    cons_ii_matrix[0, num_of_unknowns - 1] = 1
    cons_ii_b = np.array([1, ])

    # Constraint iv: p(a1 a2 b | x1 x2 z=0 y) - p(a1 a2 b | x1 x2 z=1 y) = 0   for (a1,a2,b)≠(1,1,1)   7 x 8 = 56 constraints  NOTE could also impose this by changing parameterisation
    NtoF = vs.construct_NSS_to_full_matrix_but_weird(8, 2, 8, 2)  # weirdness doesn't matter here
    cons_iii_matrix = np.zeros((2 * 56, num_of_unknowns), dtype='int')
    for (a1, a2, b), x1, x2, y in vs.cart(vs.cart(B, B, B)[:-1], B, B, B):
        cons_iii_matrix[vs.concatenate_bits(a1, a2, b, x1, x2, y)] += np.r_[NtoF[vs.concatenate_bits(a1, a2, 0, b, x1, x2, 0, y)], np.zeros(2 + dim_nss, dtype='int')]
        cons_iii_matrix[vs.concatenate_bits(a1, a2, b, x1, x2, y)] += np.r_[NtoF[vs.concatenate_bits(a1, a2, 1, b, x1, x2, 0, y)], np.zeros(2 + dim_nss, dtype='int')]
        cons_iii_matrix[vs.concatenate_bits(a1, a2, b, x1, x2, y)] -= np.r_[NtoF[vs.concatenate_bits(a1, a2, 0, b, x1, x2, 1, y)], np.zeros(2 + dim_nss, dtype='int')]
        cons_iii_matrix[vs.concatenate_bits(a1, a2, b, x1, x2, y)] -= np.r_[NtoF[vs.concatenate_bits(a1, a2, 1, b, x1, x2, 1, y)], np.zeros(2 + dim_nss, dtype='int')]

        cons_iii_matrix[56 + vs.concatenate_bits(a1, a2, b, x1, x2, y)] += np.r_[np.zeros(dim_nss + 1, dtype='int'), NtoF[vs.concatenate_bits(a1, a2, 0, b, x1, x2, 0, y)], [0]]
        cons_iii_matrix[56 + vs.concatenate_bits(a1, a2, b, x1, x2, y)] += np.r_[np.zeros(dim_nss + 1, dtype='int'), NtoF[vs.concatenate_bits(a1, a2, 1, b, x1, x2, 0, y)], [0]]
        cons_iii_matrix[56 + vs.concatenate_bits(a1, a2, b, x1, x2, y)] -= np.r_[np.zeros(dim_nss + 1, dtype='int'), NtoF[vs.concatenate_bits(a1, a2, 0, b, x1, x2, 1, y)], [0]]
        cons_iii_matrix[56 + vs.concatenate_bits(a1, a2, b, x1, x2, y)] -= np.r_[np.zeros(dim_nss + 1, dtype='int'), NtoF[vs.concatenate_bits(a1, a2, 1, b, x1, x2, 1, y)], [0]]
    cons_iii_b = np.zeros(2 * 56, dtype='int')

    # Constraint v: iv-1: p(a1=0 b λ=0 | x1 0 y) - p(a1=0 b λ=0 | x1 1 0 y) = 0  for (b,y)≠(1,1)   (6 equalities)   (set z=0 here bc everything a1a2b is independent of z anyway)
    #               iv-2: p(a2=0 b λ=1 | 0 x2 y) - p(a2=0 b λ=1 | 1 x2 0 y) = 0  for (b,y)≠(1,1)   (6 equalities)
    cons_iv_matrix = np.zeros((12, num_of_unknowns), dtype='int')
    # iv-1:  # NOTE this essentially constructs a (6,dim_nss) matrix which has as null space the 80-dim ahNSCO1 (subspace of ahNSS)
    for (b, y), x1, x2 in vs.cart(vs.cart(B, B)[:-1], B, B):
        current_row = vs.concatenate_bits(b, y, x1)
        # Find row vector that will give us p(a1=0 b λ=0 | x1 x2 y)
        # sum over a2, c
        for a2, c in vs.cart(B, B):
            cons_iv_matrix[current_row] += ((-1) ** x2) * np.r_[NtoF[vs.concatenate_bits(0, a2, c, b, x1, x2, 0, y)], np.zeros(2 + dim_nss, dtype='int')]
    # iv-2:
    for (b, y), x1, x2 in vs.cart(vs.cart(B, B)[:-1], B, B):
        current_row = 6 + vs.concatenate_bits(b, y, x2)
        # Find row vector that will give us p(a2=0 b λ=1 | x1 x2 y)
        # sum over a2, c
        for a1, c in vs.cart(B, B):
            cons_iv_matrix[current_row] += ((-1) ** x1) * np.r_[np.zeros(dim_nss + 1, dtype='int'), NtoF[vs.concatenate_bits(a1, 0, c, b, x1, x2, 0, y)], [0]]
    cons_iv_b = np.zeros(12, dtype='int')

    # The equality constraints ii-iv together:
    A_eq = np.r_[cons_ii_matrix, cons_iii_matrix, cons_iv_matrix]
    b_eq = np.r_[cons_ii_b, cons_iii_b, cons_iv_b]

    return A_ub, b_ub, A_eq, b_eq


def is_cor_in_lc(p_nss, tol=1e-12, method='highs', double_check_soln=True):
    assert len(p_nss) in [dim_nss, dim_nss + 1]
    if len(p_nss) == dim_nss + 1:
        p_nss = 1. / p_nss[-1] * np.array(p_nss[:-1])

    num_of_unknowns = (dim_nss + 1) * 2

    A_ub, b_ub, A_eq, b_eq = constraints_is_valid_lc_deco()

    # Remaining constraint: sum_λ p(...λ|..) = p_nss(...|..)
    p_nss_cons_matrix = np.concatenate((np.identity(dim_nss), np.zeros((dim_nss, 1), dtype='int'),
                                        np.identity(dim_nss), np.zeros((dim_nss, 1), dtype='int')), axis=1)
    p_nss_cons_b = p_nss

    A_eq = np.r_[A_eq, p_nss_cons_matrix]
    b_eq = np.r_[b_eq, p_nss_cons_b]

    # Do LP
    options = None if method == 'highs' else {'tol': tol}
    lp = linprog(np.zeros(num_of_unknowns, dtype='int'), A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options=options, method=method)

    # Double-check that solution is correct (if solution found)

    if double_check_soln and lp.success:
        # check that p1 := lp.x[:dim_nss + 1] is in NSCO1
        p1_nss_homog = lp.x[:dim_nss + 1]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2zy,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2) @ p1_nss_homog
            assert is_in_NSCO1z(p1_full_homog, tol)

        # check that p2 := lp.x[dim_nss + 1: 2 * dim_nss + 2] is in NSCO2
        p2_nss_homog = lp.x[dim_nss + 1: 2 * dim_nss + 2]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2) @ p2_nss_homog
            swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm_homog(lambda a1, a2, c, b, x1, x2, z, y: (a2, a1, c, b, x2, x1, z, y), num_of_binary_vars=8)
            p2_full_homog_swapped = swap_A1_A2_matrix @ p2_full_homog
            assert is_in_NSCO1z(p2_full_homog_swapped, tol)

        # check that p(λ=0) p1 + p(λ=1) p2 = p_nss
        sum_p1_p2 = p1_nss_homog[:-1] + p2_nss_homog[:-1]
        assert np.all(np.abs(sum_p1_p2 - p_nss) < tol)

    return lp


def maximum_violation_by_LC_lp(ineq, method='highs', tol=1e-12, double_check_soln=True):
    ineq = np.array(ineq)
    assert ineq.shape == (dim_nss + 1,)

    A_ub, b_ub, A_eq, b_eq = constraints_is_valid_lc_deco()

    # Objective function to minimise: -ineq @ (λp^1 + (1-λ)p^2)
    matrix = np.concatenate((np.identity(dim_nss + 1), np.identity(dim_nss + 1)), axis=1)  # multiplying the vector of unknowns with this gives the homogeneous nss representation of (λp^1 + (1-λ)p^2)
    c = -ineq @ matrix

    # Do LP
    options = None if method == 'highs' else {'tol': tol}
    lp = linprog(c, A_ub, b_ub, A_eq, b_eq, bounds=(0, 1), options=options, method=method)

    assert lp.success

    # Check solution
    if double_check_soln and lp.success:
        # check that p1 := lp.x[:dim_nss + 1] is in NSCO1
        p1_nss_homog = lp.x[:dim_nss + 1]  # this homogeneous vector represents the conditional distr p(a1a2cb|x1x2zy,λ=0)
        if p1_nss_homog[-1] != 0:  # if p(λ=0) != 0
            p1_full_homog = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2) @ p1_nss_homog
            assert is_in_NSCO1z(p1_full_homog, tol)

        # check that p2 := lp.x[dim_nss + 1: 2 * dim_nss + 2] is in NSCO2
        p2_nss_homog = lp.x[dim_nss + 1: 2 * dim_nss + 2]
        if p2_nss_homog[-1] != 0:  # if p(λ=1) != 0
            p2_full_homog = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2) @ p2_nss_homog
            swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm_homog(lambda a1, a2, c, b, x1, x2, z, y: (a2, a1, c, b, x2, x1, z, y), num_of_binary_vars=8)
            p2_full_homog_swapped = swap_A1_A2_matrix @ p2_full_homog
            assert is_in_NSCO1z(p2_full_homog_swapped, tol)

        # Check that inequality is indeed violated by lp.sol
        sum_p1_p2_h = p1_nss_homog + p2_nss_homog
        assert abs(lp.fun + ineq @ sum_p1_p2_h) < tol

    return -lp.fun


def is_in_NSCO1z(cor, tol=1e-12):
    """ cor should be given in full representation, i.e. be of length 256 or 257 """
    if len(cor) == 257:
        assert cor[-1] != 0
        cor = (1 / cor[-1]) * cor[:-1]
    else:
        assert len(cor) == 256

    # First check if in NSS. This also checks if all probs are >=0.
    if not vs.is_in_NSS(cor, 8, 2, 8, 2, tol):
        return False

    # Now check if a1a2b is independent of z
    cor = cor.reshape((2,) * 8)
    cor_a1a2b_x1x2zy = np.einsum('ijklmnop->ijlmnop', cor)
    for a1, a2, b, x1, x2, y in vs.cart(B, B, B, B, B, B):
        if abs(cor_a1a2b_x1x2zy[a1, a2, b, x1, x2, 0, y] - cor_a1a2b_x1x2zy[a1, a2, b, x1, x2, 1, y]) > tol:
            return False

    # Finally, check if a1b is independent of x2
    cor_a1b_x1x2y = np.einsum('ijklmnop->ilmnop', cor)
    for a1, b, x1, z, y in vs.cart(B, B, B, B, B):  # actually only have to check for z=0 due to previous paragraph
        if abs(cor_a1b_x1x2y[a1, b, x1, 0, z, y] - cor_a1b_x1x2y[a1, b, x1, 1, z, y]) > tol:
            return False
    return True


def make_pacb_xzy(rho_ctb, instrs_A1, instrs_A2, instrs_CT, instrs_B, dT, dB):
    """ All provided instruments should be CJ reps of instruments in the form of 4x4 matrices, NOT transposes of those (so not 'taus'). """
    # Check if all arguments are valid instruments
    for instr in np.r_[instrs_A1, instrs_A2]:
        qm.assert_is_quantum_instrument(instr, dT, dT)
    for instr in instrs_B:
        qm.assert_is_quantum_instrument(instr, dB, 1)
    for instr in instrs_CT:
        qm.assert_is_quantum_instrument(instr, 2 * dT, 1)
    qm.assert_is_quantum_instrument([rho_ctb], 1, 2 * dT * dB, num_of_outcomes=1)  # essentially checks that rho_ctb is a valid state (density matrix)

    # Make correlation. For explanation see make_pacb_xy in quantum_utils.py
    tau_dim = rho_ctb.shape[0] * instrs_A1[0][0].shape[0] * instrs_A2[0][0].shape[0] * instrs_CT[0][0].shape[0] * instrs_B[0][0].shape[0]

    # Define some index labels:
    _a1, _a2, _c, _b, _x1, _x2, _z, _y = 0, 1, 2, 3, 4, 5, 6, 7
    _CTBo, _CTBi, _A1o, _A1i, _A2o, _A2i, _CTo, _CTi, _Bo, _Bi = 8, 9, 10, 11, 12, 13, 14, 15, 16, 17

    # Construct all_taus:
    all_taus = np.einsum(rho_ctb, [_CTBo, _CTBi], instrs_A1, [_x1, _a1, _A1o, _A1i], instrs_A2, [_x2, _a2, _A2o, _A2i],
                         instrs_CT, [_z, _c, _CTo, _CTi], instrs_B, [_y, _b, _Bo, _Bi],
                         [_a1, _a2, _c, _b, _x1, _x2, _z, _y, _CTBi, _A1i, _A2i, _CTi, _Bi, _CTBo, _A1o, _A2o, _CTo, _Bo])
    all_taus2 = all_taus.reshape((2, 2, 2, 2, 2, 2, 2, 2, tau_dim, tau_dim))
    del all_taus

    # Born rule:
    pacb_xzy = np.einsum('ij,...ji->...', qm.process_operator_switch(dT, dB), all_taus2, optimize='optimal')
    del all_taus2
    if np.max(np.imag(pacb_xzy)) > 1e-15:
        print("WARNING - DETECTED A LARGE IMAGINARY VALUE IN PROBABILITY: %s" % str(np.max(np.imag(pacb_xzy))))
    return np.real(pacb_xzy)


def make_pacb_xzy_nss_h(rho_ctb, instrs_A1, instrs_A2, instrs_CT, instrs_B, dT, dB):
    return vs.construct_full_to_NSS_homog(8, 2, 8, 2) @ np.r_[make_pacb_xzy(rho_ctb, instrs_A1, instrs_A2, instrs_CT, instrs_B, dT, dB).reshape(2 ** 8), [1]]


def is_switch_cor_in_lc(rho_ctb, instrs_A1, instrs_A2, instrs_CT, instrs_B, dT, dB, tol=1e-12, method='highs'):
    cor = make_pacb_xzy_nss_h(rho_ctb, instrs_A1, instrs_A2, instrs_CT, instrs_B, dT, dB)
    if is_cor_in_lc(cor, tol=tol, method=method).success:
        return 'In LC', cor
    else:
        return 'Not in LC', cor


def chsh_rep_2222():
    ineq_full = np.zeros(17)
    for a, b, x, y in itertools.product(B, repeat=4):
        if (a + b) % 2 == x * y:
            ineq_full[vs.concatenate_bits(a, b, x, y)] += 1. / 4
    ineq_full[-1] = 0
    return vs.construct_NSS_to_full_homogeneous(2, 2, 2, 2).T @ ineq_full


def nss_var_perm_to_symm_h_2222(perm, dtype='int'):
    perm_matrix = symmetry_utils.full_perm_to_symm_homog(perm, num_of_binary_vars=4, dtype=dtype)
    return vs.construct_full_to_NSS_homog(2, 2, 2, 2) @ perm_matrix @ vs.construct_NSS_to_full_homogeneous(2, 2, 2, 2)


def bell_2222_symmetries():
    return [
        nss_var_perm_to_symm_h_2222(lambda a, b, x, y: (a, b, x, (y + 1) % 2)),
        nss_var_perm_to_symm_h_2222(lambda a, b, x, y: (a, b, (x + 1) % 2, y)),
        nss_var_perm_to_symm_h_2222(lambda a, b, x, y: (a, (b + y) % 2, x, y)),
        nss_var_perm_to_symm_h_2222(lambda a, b, x, y: ((a + x) % 2, b, x, y)),
        nss_var_perm_to_symm_h_2222(lambda a, b, x, y: (b, a, y, x)),
    ]


def print_chsh_violations(p_acbxzy_nss):
    p_full = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2) @ p_acbxzy_nss
    p_full = (1 / p_full[-1] * p_full[:-1]).reshape((2,) * 8)
    bell_facets = utils.read_vertex_range_from_file('panda-files/chsh_2222')
    p_cbxzy = np.einsum('aecbxwzy->cbxwzy', p_full)
    for x1, x2 in vs.cart(B, B):
        cor = np.r_[vs.construct_full_to_NSS_matrix(2, 2, 2, 2) @ p_cbxzy[:, :, x1, x2, :, :].reshape(16), [1]]
        print('CHSH violation for x1=%d, x2=%d: %s' % (x1, x2, str(utils.max_violation_h(cor, bell_facets))))
    p_cbzy_avg = 1 / 4 * (p_cbxzy[:, :, 0, 0, :, :] + p_cbxzy[:, :, 0, 1, :, :] + p_cbxzy[:, :, 1, 0, :, :] + p_cbxzy[:, :, 1, 1, :, :])
    p_cbzy_avg = np.r_[vs.construct_full_to_NSS_matrix(2, 2, 2, 2) @ p_cbzy_avg.reshape(16), [1]]
    print('CHSH violation of averaged correlation: %s' % str(utils.max_violation_h(p_cbzy_avg, bell_facets)))


def ineq1():  # TODO is this one violated by the switch too?
    """ The first violated LC inequality I found, and which I wrote in the note to Giulio and Jon. """
    ineq_full = np.zeros((2,) * 8)
    for a1, a2, c, b, x1, x2, z, y in itertools.product((0, 1), repeat=8):
        if b == 0 and a2 == x1 and y == 0:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 8
        if b == 1 and a1 == x2 and y == 0:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 8
        if (b + c) % 2 == y * z and x1 == x2 == 0:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 4
    ineq_full_h = np.r_[ineq_full.flatten(), [0]]
    ineq_nss_h = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2).T @ ineq_full_h
    return ineq_nss_h


def ineq2():
    ineq_full = np.zeros((2,) * 8)
    for a1, a2, c, b, x1, x2, z, y in itertools.product((0, 1), repeat=8):
        if b == 0 and a2 == x1 and y == 0:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 8
        if b == 1 and a1 == x2 and y == 0:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 8
        if (b + c) % 2 == ((y + 1) % 2) * z and x1 == x2 == 0:  # due to some symmetry inconvenience, this one (with y+1) is violated by the switch cor described in my note.
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 4
    ineq_full_h = np.r_[ineq_full.flatten(), [0]]
    ineq_nss_h = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2).T @ ineq_full_h
    return ineq_nss_h


def ineq_beta_gamma_delta(beta, gamma, delta):
    """ See note """
    ineq_full = np.zeros((2,) * 8)
    for a1, a2, c, b, x1, x2, z, y in itertools.product((0, 1), repeat=8):
        if b == 0 and a2 == (x1 + beta * z) % 2 and y == delta:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 8
        if b == 1 and a1 == (x2 + gamma * z) % 2 and y == delta:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 8
        if (b + c) % 2 == y * z and x1 == x2 == 0:
            ineq_full[a1, a2, c, b, x1, x2, z, y] += 1 / 4
    ineq_full_h = np.r_[ineq_full.flatten(), [0]]
    ineq_nss_h = vs.construct_NSS_to_full_homogeneous(8, 2, 8, 2).T @ ineq_full_h
    return ineq_nss_h


def construct_vertices():
    # Construct marginalisation map Sigma_c : LC_acbxy -> LC_abxy
    marg_map_f_h = np.zeros((65, 129), dtype='int8')
    for a1, a2, b, x1, x2, y in itertools.product(B, repeat=6):
        marg_map_f_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 0, b, x1, x2, y)] = 1
        marg_map_f_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 1, b, x1, x2, y)] = 1
    marg_map_f_h[-1][-1] = 1
    marg_map_n_h = vs.construct_full_to_NSS_homog(4, 2, 4, 2) @ marg_map_f_h @ vs.construct_NSS_to_full_homogeneous(8, 2, 4, 2)
    # TODO test this map

    print('Loading LC_acbxy vertices...')
    lc_acbxy_vertices = np.load('panda-files/results/lc_vertices.npy')
    num_of_vertices = len(lc_acbxy_vertices)

    print('Marginalising LC_acbxy vertices...')
    lc_acbxy_vertices_marged = lc_acbxy_vertices @ marg_map_n_h.T  # = (marg_map_n_h @ lc_acbxy_vertices.T).T
    del lc_acbxy_vertices
    # print('Normalising marginalised vertices...')
    # for i in range(len(lc_acbxy_vertices_marged)):
    #     gcd = functools.reduce(math.gcd, lc_acbxy_vertices_marged[i])
    #     if gcd != 1:
    #         print('Row %d was not normalised' % i)  NOTE this was never called, so normalising is unnecessary
    #         lc_acbxy_vertices_marged[i] = 1/gcd * lc_acbxy_vertices_marged[i]
    print('Partioning marginalised vertices...')
    V = []  # an array of arrays; each inner array consists of numbers that represent indices of equal rows of lc_acbxy_vertices_marged
    for j in range(num_of_vertices):
        vj = lc_acbxy_vertices_marged[j]
        j_was_put_in_a_V_i = False
        for V_i in V:
            vi = lc_acbxy_vertices_marged[V_i[0]]
            # Check if all(vi == vj)
            vi_is_vj = True
            for k in range(len(vj)):
                if vi[k] != vj[k]:
                    vi_is_vj = False
                    break
            if vi_is_vj:
                V_i.append(j)
                j_was_put_in_a_V_i = True
                break

        if not j_was_put_in_a_V_i:
            V.append([j])
        if j % 50000 == 0:
            print('j = %d, len(V) = %d' % (j, len(V)))
    del lc_acbxy_vertices_marged

    # Double-check that V is a partition of range(lc_acbxy_vertices)
    if np.all(sorted([item for sublist in V for item in sublist]) == list(range(num_of_vertices))):
        print('V is a partition of range(num_of_vertices)')
    else:
        print('Error: V is not a valid partition!')

    print('len(V)=%d' % len(V))
    print('num of vertices=%d' % num_of_vertices)
    print('maximum length of element of V is %d' % max([len(V_i) for V_i in V]))
    print('minimum length of element of V is %d' % min([len(V_i) for V_i in V]))
    print('Saving V...')
    with open('V', 'w') as f:
        for V_i in V:
            f.write(' '.join(map(str, [i for i in V_i])) + '\n')


if __name__ == '__main__':
    ## Demonstrating violation of LC:
    result, cor = is_switch_cor_in_lc(method='highs',
                                      rho_ctb=qm.rho_tcb_0phi,
                                      instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                      instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                      instrs_CT=[qm.instr_C_to_instr_CT(qm.instr_vn_destr(qm.z_onb)), qm.instr_C_to_instr_CT(qm.instr_vn_destr(qm.onb_from_direction(.25 * np.pi)))],
                                      instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.onb_from_direction(.25 * np.pi))],  # TODO think about why this violates LC!
                                      dT=2, dB=2)
    print(result)
    print_chsh_violations(cor)

    ## Checking my analytical calculations about the inequality ineq1()
    """
    ineq = ineq2()
    print(lp_max_violation_by_LC(ineq))  # Indeed gives 7/4!
    print(ineq @ make_pacb_xzy_nss_h(rho_ctb=qm.rho_tcb_0phi,
                                        instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                        instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                        instrs_CT=[qm.instr_C_to_instr_CT(qm.instr_vn_destr(qm.diag1_onb)), qm.instr_C_to_instr_CT(qm.instr_vn_destr(qm.diag2_onb))],
                                        instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.x_onb)],
                                        dT=2, dB=2))  # Indeed gives 3/2 + 1/(2sqrt2) !
    """

    ## Computing maximal LC value for the 'beta,gamma,delta' inequalities (see note)
    """
    for beta, gamma, delta in itertools.product(B, repeat=3):
        print('%d,%d,%d: %s' % (beta, gamma, delta, str(lp_max_violation_by_LC(ineq_beta_gamma_delta(beta, gamma, delta)))))
    """

    # construct_vertices()

    # result, cor = is_switch_cor_in_lc(method='interior-point', tol=1e-8,
    #                                   rho_ctb=qm.rho_ctb_plusphiplus,
    #                                   instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
    #                                   instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
    #                                   instrs_CT=[qm.instr_C_to_instr_CT(qm.instr_vn_destr(qm.x_onb)), qm.instr_C_to_instr_CT(qm.instr_vn_destr(qm.x_onb))],
    #                                   instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)],
    #                                   dT=2, dB=2)
    # print(result)
