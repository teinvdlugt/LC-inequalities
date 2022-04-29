import cmath
import functools
import itertools
import random
import sys
from math import pi, cos, sin, sqrt
import numpy as np

import scipy.linalg

import lp_for_membership
import symmetry_utils
import towards_lc
import utils
import vector_space_utils as vs


## PROCESS OPERATORS
def process_operator_switch(dT=2, dB=2):
    """ Returns process operator of a switch on Alice's side and just an incoming system on Bob's side.
    Here the rho_ctb state is not yet plugged in, so the result includes tensor factors corresponding to the systems Cout*, Tout*, Bout*.
    The order of tensor factors is:
            (Cout*, Tout*, Bout*, A1in, A1out*, A2in, A2out*, C~in, T~in, B~in).
    :param dT: The dimension of the target system fed into the switch.
    :param dB: The dimension of Bob's system.
    """
    # Outline;
    #  - Construct CO1 part of W (this is a vector, not an operator)
    #  - Construct CO2 part of W
    #  - Make sure that the order of tensor factors in both parts are equal to the one promised above
    #  - Add the parts together to get W
    #  - return the matrix |W><W| = proj(W)
    # Work with row vectors, i.e. of shape (d,), because those work with our reshuffling function.

    phi_plus_dT = np.identity(dT)  # NOTE: unnormalised phi+!
    phi_plus_dB = np.identity(dB)

    # CO1 part of W. See [p72].  Tensor order before reordering of the Einstein indices: Bout*, B~in, Cout*, C~in, Tout*, A1in, A1out*, A2in, A2out*, T~in
    W_CO1 = np.einsum('ij,k,l,mn,op,qr->kminopqlrj', phi_plus_dB, ket0, ket0, phi_plus_dT, phi_plus_dT, phi_plus_dT).reshape((dB ** 2) * 2 * 2 * (dT ** 6))

    # CO2 part of W. See [p72].  Tensor order before reordering of the Einstein indices: Bout*, B~in, Cout*, C~in, Tout*, A2in, A2out*, A1in, A1out*, T~in
    W_CO2 = np.einsum('ij,k,l,mn,op,qr->kmipqnolrj', phi_plus_dB, ket1, ket1, phi_plus_dT, phi_plus_dT, phi_plus_dT).reshape((dB ** 2) * 2 * 2 * (dT ** 6))
    # (Note: the only difference is that ket0 becomes ket1 for C and that A1 and A2, which correspond to no and pq, are interchanged.

    # assert np.all(np.einsum('minopqrj->mipqnorj', W_CO1[0, :, :, :, :, :, :, 0, :, :]) == W_CO2[1, :, :, :, :, :, :, 1, :, :])
    # assert np.all(np.einsum('minopqrj->mipqnorj', (W_CO1 + W_CO2)[0, :, :, :, :, :, :, 0, :, :]) == (W_CO1 + W_CO2)[1, :, :, :, :, :, :, 1, :, :])

    # Add together and return projection (which is the process operator)
    return proj(W_CO1 + W_CO2)


def make_pacb_xy_noTmmt(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, dT, dB):
    # instr_CT should be Ci ⊗ Ti (not ⊗Co⊗To, because output systems are trivial), with Ci ⊗ Co in (CJ) state
    # instr_C and Ti in state np.identity(dT).
    instr_CT = kron(instr_C, np.identity(dT))
    return make_pacb_xy(rho_ctb, instrs_A1, instrs_A2, instr_CT, instrs_B, dT, dB)


def make_pacb_xy(rho_ctb, instrs_A1, instrs_A2, instr_CT, instrs_B, dT, dB):
    """ All provided instruments should be CJ reps of instruments in the form of 4x4 matrices, NOT transposes of those (so not 'taus'). """
    # Check if all arguments are valid instruments
    for instr in np.r_[instrs_A1, instrs_A2]:
        assert_is_quantum_instrument(instr, dT, dT)
    for instr in instrs_B:
        assert_is_quantum_instrument(instr, dB, 1)
    assert_is_quantum_instrument(instr_CT, 2 * dT, 1)
    assert_is_quantum_instrument([rho_ctb], 1, 2 * dT * dB, num_of_outcomes=1)  # essentially checks that rho_ctb is a valid state (density matrix)

    # Make correlation
    # Bundle all possible '⊗tau's for all settings and outcomes together into one tensor 'all_taus':
    # all_taus[a1,a2,c,b,x1,x2,y] should give be the same as kron(rho_ctb.T, instrs_A1[x1][a1].T, instrs_A2[x2][a2].T, instr_C[c].T, tau_Ttilde, instrs_B[y][b].T)
    # To get kron of matrices, move all 'output' indices to the front and all 'input' indices to the back, and then reshape.
    # e.g.: kron(A,B,C) == np.einsum('ij,kl,mn->ikmjln', A,B,C).reshape((A.shape[0]*B.shape[0]*C.shape[0], A.shape[1]*B.shape[1]*C.shape[1]))
    # so that kron(A.T,B.T,C.T) == np.einsum('ij,kl,mn->jlnikm', A,B,C).reshape((A.shape[1]*B.shape[1]*C.shape[1], A.shape[0]*B.shape[0]*C.shape[0]))
    # Here 'input' and 'output' refer to the CJ reps seen as operators on a space, so NOT to input and output systems in the quantum sense.

    # For given settings and outcomes, ⊗tau is an operator (square matrix) on the space with the following dimension:
    tau_dim = rho_ctb.shape[0] * instrs_A1[0][0].shape[0] * instrs_A2[0][0].shape[0] * instr_CT[0].shape[0] * instrs_B[0][0].shape[0]

    # Define some index labels:
    _a1, _a2, _c, _b, _x1, _x2, _y = 0, 1, 2, 3, 4, 5, 6
    _CTBo, _CTBi, _A1o, _A1i, _A2o, _A2i, _CTo, _CTi, _Bo, _Bi = 7, 8, 9, 10, 11, 12, 13, 14, 15, 16

    # Construct all_taus:
    all_taus = np.einsum(rho_ctb, [_CTBo, _CTBi], instrs_A1, [_x1, _a1, _A1o, _A1i], instrs_A2, [_x2, _a2, _A2o, _A2i],
                         instr_CT, [_c, _CTo, _CTi], instrs_B, [_y, _b, _Bo, _Bi],
                         [_a1, _a2, _c, _b, _x1, _x2, _y, _CTBi, _A1i, _A2i, _CTi, _Bi, _CTBo, _A1o, _A2o, _CTo, _Bo])
    # assert np.all(all_taus == np.einsum(all_taus, [_a1, _a2, _c, _b, _x1, _x2, _y, _CTBi, _A1i, _A2i, _CTi, _Bi, _CTBo, _A1o, _A2o, _CTo, _Bo],
    #                              [_a2, _a1, _c, _b, _x2, _x1, _y, _CTBi, _A2i, _A1i, _CTi, _Bi, _CTBo, _A2o, _A1o, _CTo, _Bo]))  # to check if symmetric in A1 <-> A2.
    all_taus = all_taus.reshape((2, 2, 2, 2, 2, 2, 2, tau_dim, tau_dim))

    # Born rule:
    pacb_xy = np.einsum('ij,...ji->...', process_operator_switch(dT, dB), all_taus, optimize='optimal')  # This is trace(matmul(proc_op, all_taus)): 'ij,...jk->...ik' followed by '...ii->...'
    del all_taus
    if np.max(np.imag(pacb_xy)) > 1e-15:
        print("WARNING - DETECTED A LARGE IMAGINARY VALUE IN PROBABILITY: %s" % str(np.max(np.imag(pacb_xy))))
    pacb_xy = np.real(pacb_xy)

    # Old, more legible code:
    """
    proc_op = process_operator_switch(dT, dB)
    pacb_xy = np.zeros((2,) * 7)
    tau_Ttilde = np.identity(dT)  # trace out the output target system
    for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
        taus = kron(rho_ctb.T, instrs_A1[x1][a1].T, instrs_A2[x2][a2].T, instr_CT[c].T, instrs_B[y][b].T)
        born_prob = np.einsum('ij,ji', proc_op, taus)
        if np.imag(born_prob) > 1e-15:
            print("WARNING - DETECTED A LARGE IMAGINARY VALUE IN PROBABILITY: p(%d,%d,b=%d,c=%d,%d,%d,%d) = %s" % (a1, a2, b, c, x1, x2, y, str(born_prob)))
        pacb_xy[a1, a2, c, b, x1, x2, y] = np.real(born_prob)
    """

    return pacb_xy


def assert_is_quantum_instrument(instr, d_in, d_out, num_of_outcomes=2, tol=1e-12):
    """ Checks whether instr is a CJ rep of a valid quantum instrument from a d_in-dinmnsional system
    to a d_out-dimensional system. That is, that for any i, instr[i] is a CJ rep of a CP map, such
    that all instr[i] together sum to a CPTP map. CJ reps are done in the basis-independent way, viz. taking
    the dual of the input space.
    TODO treat H_out ⊗ H_in* as tensor or kronecker product? """
    assert len(instr) == num_of_outcomes
    # Check that all maps are of the right dimension and are CP
    for _map in instr:
        # assert _map.shape == (d_out, d_out, d_in, d_in)  # operator on H_in* ⊗ H_out
        assert _map.shape == (d_in * d_out, d_in * d_out)
        # CP is equivalent to the CJ rep being positive semidefinite as an operator (d_out*d_in, d_out*d_in)
        # _map_as_kron_matrix = tns_to_kron_of_ops(_map)  # NOTE these 'out' and 'in' are not the same as in tns_to_kron_of_ops
        assert np.all(np.abs(_map - _map.conj().T) < tol)  # finding eigenvalues only works when self-adjoint. TODO right?
        assert np.all(np.linalg.eigvals(_map) > -tol)

    # CPTP is equivalent to Tr_out(CJ rep) == Id_in
    sum_of_maps = np.sum(instr, axis=0)
    sum_tns = sum_of_maps.reshape((d_in, d_out, d_in, d_out))  # tensor rather than kronecker rep
    assert np.all(np.abs(np.trace(sum_tns, axis1=1, axis2=3) - np.identity(d_in)) < tol)


def quantum_cor_nss_noTmmt(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, dT=2, dB=2, common_multiple_of_denominators=None):
    cor_full = make_pacb_xy_noTmmt(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, dT, dB).reshape((128,))
    return vs.full_acb_to_nss_homog(cor_full, common_multiple_of_denominators)


def quantum_cor_nss(rho_ctb, instrs_A1, instrs_A2, instr_CT, instrs_B, dT=2, dB=2, common_multiple_of_denominators=None):
    cor_full = make_pacb_xy(rho_ctb, instrs_A1, instrs_A2, instr_CT, instrs_B, dT, dB).reshape((128,))
    return vs.full_acb_to_nss_homog(cor_full, common_multiple_of_denominators)


def quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb, X1, X2, c_onb, Y, common_multiple_of_denominators=None):
    return quantum_cor_nss_noTmmt(rho_ctb=rho_ctb,
                                  instrs_A1=np.array([instr_vn_nondestr(onb) for onb in X1]),
                                  instrs_A2=np.array([instr_vn_nondestr(onb) for onb in X2]),
                                  instr_C=instr_vn_destr(c_onb),
                                  instrs_B=np.array([instr_vn_destr(onb) for onb in Y]),
                                  dT=2, dB=2, common_multiple_of_denominators=common_multiple_of_denominators)


def random_quantum_setup_qubit_vn_noTmmt():
    return random_pure_density_matrix(allow_complex=True), \
           [instr_vn_nondestr(random_onb()), instr_vn_nondestr(random_onb())], \
           [instr_vn_nondestr(random_onb()), instr_vn_nondestr(random_onb())], \
           instr_vn_destr(random_onb()), \
           [instr_vn_destr(random_onb()), instr_vn_destr(random_onb())]


def random_quantum_setup_qubit_vn():
    rho_ctb = random_pure_density_matrix()
    instrs_A1 = np.array([instr_proj_mmt_nondestr(random_ket()), instr_proj_mmt_nondestr(random_ket())])
    instrs_A2 = np.array([instr_proj_mmt_nondestr(random_ket()), instr_proj_mmt_nondestr(random_ket())])
    # For the mmt on CT, take a projective measurement, either with projections of rank 2 and 2 or with projections of rank 1 and 3.
    ct_proj_rank = np.random.randint(1, 3)
    ct_proj = random_orth_proj(4, ct_proj_rank)
    instr_CT = np.array([nondestr_cj_to_destr_cj(kraus_op_to_cj(ct_proj)), nondestr_cj_to_destr_cj(kraus_op_to_cj(np.identity(4) - ct_proj))])
    instrs_B = np.array([instr_proj_mmt_destr(random_ket()), instr_proj_mmt_destr(random_ket())])
    return rho_ctb, instrs_A1, instrs_A2, instr_CT, instrs_B, 2, 2


def somewhat_random_qubit_instr_nondestr():
    return np.random.choice([
        instr_proj_mmt_nondestr(random_ket()),
        instr_measure_and_send_fixed_state(random_onb(), random_ket()),
        instr_do_nothing,
        np.array([kraus_op_to_cj(random_unitary(dim=2)), kraus_op_to_cj(np.zeros(2, 2))]),
        instr_weak_vN_nondestr(random_ket(), noise=np.random.rand())
    ])


def random_quantum_setup_qutrit_proj():
    """ Target system qutrit, Bob qubit """
    rho_ctb = random_pure_density_matrix(dim=12)
    instrs_A1 = np.array([instr_proj_mmt_nondestr(random_ket(3)), instr_proj_mmt_nondestr(random_ket(3))])
    instrs_A2 = np.array([instr_proj_mmt_nondestr(random_ket(3)), instr_proj_mmt_nondestr(random_ket(3))])

    # For the mmt on CT, take a projective measurement, the first projection of which has rank 1,2 or 3 (CT has dim 6 so this covers all proj mmts).
    ct_proj_rank = np.random.randint(1, 4)
    ct_proj = random_orth_proj(6, ct_proj_rank)
    instr_CT = np.array([nondestr_cj_to_destr_cj(kraus_op_to_cj(ct_proj)), nondestr_cj_to_destr_cj(kraus_op_to_cj(np.identity(6) - ct_proj))])

    instrs_B = np.array([instr_proj_mmt_destr(random_ket()), instr_proj_mmt_destr(random_ket())])
    dT = 3
    dB = 2
    return rho_ctb, instrs_A1, instrs_A2, instr_CT, instrs_B, dT, dB


def quantum_setup_two_channels():
    """ Returns the quantum instruments corresponding to the 'two channels' scenario, where the target system is
     two qubits, each of which is used for a communication direction (i.e. A1->A2 and A2->A1, respectively).
     Precisely: A1 measures target qubit 2 in computational basis and prepares target qubit 1 in computational basis state according to setting x1
                A2 measures target qubit 1 in computational basis and prepares target qubit 2 in computational basis state according to setting x2
                B measures his (single) qubit in Z or X basis depending on setting y
                C measures control qubit in X basis
    TODO generalise this?
                no initial state is returned by this function
    :returns instrs_A1, instrs_A2, instr_C, instrs_B
    """
    ## instrs_A1: prepare qubit 1 in |x1> and measure qubit 2
    instrs_A1 = np.zeros((2, 2, 16, 16))  # 2 settings, 2 outcomes, and CP maps are 16x16 matrices
    for x1 in (0, 1):
        x1_ket = [ket0, ket1][x1]
        instrs_A1[x1] = [kron(np.identity(2), proj(ket0), proj(x1_ket), proj(ket0)),  # TODO also wrong because of missing conjugate?
                         kron(np.identity(2), proj(ket1), proj(x1_ket), proj(ket1))]
    ## instrs_A2: prepare qubit 2 in |x2> and measure qubit 1
    instrs_A2 = np.zeros((2, 2, 16, 16))  # 2 settings, 2 outcomes, and CP maps are 16x16 matrices
    for x2 in (0, 1):
        x2_ket = [ket0, ket1][x2]
        instrs_A2[x2] = [kron(proj(ket0), np.identity(2), proj(ket0), proj(x1_ket)),
                         kron(proj(ket1), np.identity(2), proj(ket1), proj(x1_ket))]
    ## instrs_B: measure in Z or X basis
    instrs_B = [instr_vn_destr(z_onb), instr_vn_destr(x_onb)]
    ## instr_C: measure in X basis
    instr_C = instr_vn_destr(x_onb)

    return instrs_A1, instrs_A2, instr_C, instrs_B


def generate_some_quantum_cors_complete_vn_noTmmt(file_to_save_to=None, num_of_random_cors=1000, common_multiple_of_denominators=None):
    """
    If common_multiple_of_denominators is not None, then the generated correlations are approximated by fractions with
    the given denominator.
    """
    qm_cors = [
        quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb=rho_ctb_plusphiplus, X1=[z_onb, x_onb], X2=[z_onb, x_onb], c_onb=x_onb, Y=[z_onb, x_onb],
                                                     common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb=rho_tcb_0phi, X1=[z_onb, x_onb], X2=[z_onb, x_onb], c_onb=x_onb, Y=[diag1_onb, diag2_onb],
                                                     common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb=rho_tcb_0phi, X1=[z_onb, x_onb], X2=[z_onb, x_onb], c_onb=diag1_onb, Y=[z_onb, x_onb],
                                                     common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb=rho_tcb_0phi, X1=[diag1_onb, diag2_onb], X2=[z_onb, x_onb], c_onb=diag2_onb, Y=[z_onb, x_onb],
                                                     common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb=rho_ctb_ghz, X1=[diag1_onb, diag2_onb], X2=[diag1_onb, diag2_onb], c_onb=x_onb, Y=[diag1_onb, diag2_onb],
                                                     common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb=rho_ctb_ghz, X1=[z_onb, x_onb], X2=[z_onb, x_onb], c_onb=diag1_onb, Y=[z_onb, x_onb],
                                                     common_multiple_of_denominators=common_multiple_of_denominators)]
    if file_to_save_to is not None:
        np.save(file_to_save_to, qm_cors)
    for i in range(0, num_of_random_cors):
        print("Generated %d / %d random qm cors" % (i, num_of_random_cors))
        qm_cors.append(quantum_cor_nss_noTmmt(*random_quantum_setup_qubit_vn_noTmmt(), common_multiple_of_denominators=common_multiple_of_denominators))
        if file_to_save_to is not None:
            np.save(file_to_save_to, qm_cors)
    return qm_cors


def generate_some_quantum_cors_complete_vn(file_to_save_to=None, num_of_random_cors=1000, common_multiple_of_denominators=None):
    """
    If common_multiple_of_denominators is not None, then the generated correlations are approximated by fractions with
    the given denominator.
    """
    qm_cors = []
    for i in range(0, num_of_random_cors):
        print("Generated %d / %d random qm cors" % (i, num_of_random_cors))
        qm_cors.append(quantum_cor_nss(*random_quantum_setup_qubit_vn(), common_multiple_of_denominators=common_multiple_of_denominators))
        if file_to_save_to is not None:
            np.save(file_to_save_to, qm_cors)
    return qm_cors


def generate_some_proj_qutrit_cors(file=None, num_of_random_cors=1000):
    qm_cors = []
    for i in range(num_of_random_cors):
        print("Generated %d / %d random qm cors" % (i, num_of_random_cors))
        sys.stdout.flush()
        qm_cors.append(quantum_cor_nss(*random_quantum_setup_qutrit_proj()))
        if file is not None:
            np.save(file, qm_cors)


def some_quantum_violations(ineq, quantum_cor_file='panda-files/some_quantum_cors3.npy'):
    """ Tests some sensibly & randomly generated quantum correlations against the provided inequality and returns the largest violation found. """
    # qm_cors2 = utils.read_vertex_range_from_file('panda-files/some_quantum_cors2', dtype='float64')
    qm_cors = np.load(quantum_cor_file).astype('float64')

    violations = np.matmul(ineq, qm_cors.T)
    for i in range(0, len(qm_cors)):
        violations[i] /= float(qm_cors[i][-1])
    return violations


## GENERAL UTILITY FUNCTIONS
def proj(vector):
    """ Returns the matrix representing projection onto the given vector.
     The vector should be a ROW vector, i.e. of shape (d,) """
    return np.einsum('i,j->ij', vector, vector.conj())  # Same as vector.reshape((n,1)) @ vector.reshape((1,n)).conj()


def kron(*factors):
    """
    :param factors: list of numpy matrices
    :return: consecutive Kronecker product of the matrices, in the order as appearing in factors.
    """
    """
    if len(factors) == 0:
        raise ValueError()
    if len(factors) == 1:
        return factors[0]

    product = np.kron(factors[0], factors[1])
    for i in range(2, len(factors)):
        product = np.kron(product, factors[i])

    return product
    """

    # Later learned that this can also be done easier as follows:
    return functools.reduce(np.kron, factors)


def reshuffle_kron(matrix, perm):  # TODO doesn't work yet. For now use reshuffle_kron_vector()
    """
    :param matrix: a np.array which is a kronecker product of n := len(perm) 2x2 matrices, i.e. it is a 2^n x 2^n matrix.
    :param perm: the permutation along which to shuffle the Kronecker factors, provided as a length-n list;
        e.g. (1,0,3,2) swaps the order of the first two and the last two factors.
    :return The permuted Kronecker product. Example: reshuffle(kron(A,B,C,D), (0,1,3,2)) gives the same as kron(B,A,D,C).
        It is crucial that all matrices A,B,C,D are assumed to be 2-dimensional. But you can also use this function for
        power-of-2 dimensional matrices: e.g. if A,B are two-dimensional and C is four-dimensional, then
        reshuffle(kron(A,B,C), (0,2,3,1)) gives the same as kron(A,C,B).
    """
    print("The function reshuffle_kron() which you're using doesn't work!")
    # Based on https://stackoverflow.com/a/50889816/4129739. Yet to understand!
    # Example (from that SO answer): if order is ABCD and you want CADB, then perm is (2, 0, 3, 1).
    # According to the example (which is also about 2x2 matrices I believe), you then need (after reshaping) to
    # transpose along the tuple (2, 0, 3, 1, 2+4, 0+4, 3+4, 1+4) = (2, 0, 3, 1, 6, 4, 7, 5)
    # So let's generalise without actually understanding what happens
    n = len(perm)
    # (n-1,n-1,n-1,n-1) - (2,0,3,1) = (n-1-2, n-1-0, n-1-3, n-1-1) = (1, 3, 0, 2) =: x,
    # and then flip it the other way around to get (2,0,3,1) =: x_flip (in this case the same as what we started with)
    x = np.array([n - 1, ] * n) - perm
    x_flip = [x[i] for i in np.arange(n - 1, -1, step=-1)]
    transpose_tuple = np.r_[x_flip, x_flip + n * np.ones(n)].astype('int32')
    print(transpose_tuple)
    return matrix.reshape([2, ] * 2 * n).transpose(transpose_tuple).reshape(2 ** n, 2 ** n)


def reshuffle_kron_vector(kron_vector, perm):
    """
    :param kron_vector: a ROW vector, i.e. of shape (d,), which is a kronecker product of n := len(perm) 2d vectors, i.e. d = 2^n.
    :param perm: the permutation along which to shuffle the Kronecker factors, provided as a length-n list;
        e.g. (1,0,3,2) swaps the order of the first two and the last two factors.
    :return The permuted Kronecker product. Example: reshuffle(kron(A,B,C,D), (0,1,3,2)) gives the same as kron(B,A,D,C).
        It is crucial that all vectors A,B,C,D are assumed to be of length 2. But you can also use this function for
        power-of-2-length vectors: e.g. if A,B are two-dimensional and C is four-dimensional, then
        reshuffle(kron(A,B,C), (0,2,1)) gives the same as kron(A,C,B). TODO claim sentence is wrong?

    For some reason, some small error is generated. TODO fix if/when necessary, because rounding errors shouldn't play a role here
    (since the only thing that happens should be reordering some elements of a vector).

    Approach inspired by https://stackoverflow.com/questions/50883773/permute-string-of-kronecker-products/50889816
    """
    # For vectors (rather than general matrices) the correspondence between kronecker and tensor product is easier,
    # and can be done using just reshaping.
    n = len(perm)  # number of factors in the kronecker/tensor product
    if kron_vector.shape != (2 ** n,):
        raise ValueError('kron_vector should be of shape (2**len(perm),)')
    # Convert to tensor product:
    tens_vector = kron_vector.copy().reshape((2,) * n)  # tested
    # Do the permutation:
    tens_vector = tens_vector.transpose(perm)
    # Switch back to kronecker product:
    return tens_vector.reshape((2 ** n,))  # tested


def tns_to_kron_of_ops(tns):
    """ Convert between ways of describing an operator H1in ⊗ H2in -> H1out ⊗ H2out.
    :param tns: A 4-tensor with dimensions (d1out, d1in, d2out, d2in).
    :return: A 2-tensor with dimensions (d1out * d2out, d1in * d2in).
    """
    d1out, d1in, d2out, d2in = tns.shape
    # tns is an element of B(H1in, H1out) ⊗ B(H2in, H2out)
    # so it is a finite sum of product elements A1 ⊗ A2,   Ai in B(Hiin, Hiout).
    # if I can get the product elts then I just need to take their kron and add together.
    # How to get product elements? In an easier scenario: how to get product elts of an a in H ⊗ K?
    # a is a 2-tensor with dimensions (dH, dK). a = sum_ij a_ij |i>_H ⊗ |j>_K
    # So the product elements are a_ij |i>_H ⊗ |j>_K == a[i,j] δij with δij the 2-tensor that only
    # has a 1 at indices [i,j].
    """
    result = np.zeros((d1out * d2out, d1in * d2in))
    original = np.zeros_like(tns)
    for i, j, k, l in itertools.product(range(0, d1out), range(0, d1in), range(0, d2out), range(0, d2in)):
        coeff = tns[i, j, k, l]
        tns_factor1 = np.zeros((d1out, d1in))
        tns_factor1[i, j] = 1
        tns_factor2 = np.zeros((d2out, d2in))
        tns_factor2[k, l] = 1
        tensor_prod = np.einsum('ij,kl->ijkl', tns_factor1, tns_factor2)
        assert np.all(np.argwhere(tensor_prod) == (i, j, k, l))
        kron_prod = kron(tns_factor1, tns_factor2)
        assert np.all(np.argwhere(kron_prod) == (i * d2out + k, j * d2in + l))
        assert np.all(tensor_prod.swapaxes(1,2).reshape((d1out * d2out, d1in * d2in)) == kron_prod)
        original += coeff * tensor_prod
        result += coeff * kron_prod
    assert np.all(original == tns)
    assert np.all(result == tns.swapaxes(1,2).reshape((d1out * d2out, d1in * d2in)))
    return result
    """
    return tns.swapaxes(1, 2).reshape((d1out * d2out, d1in * d2in))


def complex_exp(phi):
    """
    :param phi: exponent
    :returns e^{i phi}, but not a complex number if not necessary
    """
    if phi == 0:
        return 1
    if phi == pi / 2:
        return 1j
    if phi == pi:
        return -1
    if phi == 3 * pi / 2:
        return -1j
    return cmath.exp(1j * phi)


def normalise_vec(vector):
    return 1 / np.linalg.norm(vector) * vector


def random_pure_density_matrix(dim=8, allow_complex=True):
    return proj(random_ket(dim, allow_complex))


def random_ket(dim=2, allow_complex=True):
    if allow_complex:
        return normalise_vec(np.random.rand(dim) + np.random.rand(dim) * 1j)
    else:
        return normalise_vec(np.random.rand(dim))


def onb_from_direction(theta, phi=0.):
    """ Returns the mmt ONB corresponding to measurement of a spin in the specified direction, using the Bloch representation.
     :param theta: polar angle of direction, in (0,π)
     :param phi: azimuthal angle of direction, in (0,2π)
     :return a tuple (v1, v2) of orthogonal, normalised 2d complex vectors.
    """
    # The other vector will be given by the opposite direction in the Bloch sphere:
    theta_opp, phi_opp = pi - theta, phi + pi  # TODO why does this not work? Why don't I get an orthogonal state?
    # Pick out some special cases to reduce error. Because python thinks cos(pi/2) = 6.123234e-17
    cos_theta_half = 0 if theta == pi else cos(theta / 2.)
    cos_theta_opp_half = 0 if theta_opp == pi else cos(theta_opp / 2.)
    sin_theta_half = sin(theta / 2.)  # No such problems for sin on the interval (0,pi/2)
    sin_theta_opp_half = sin(theta_opp / 2.)

    return np.array([[cos_theta_half, sin_theta_half * complex_exp(phi)],
                     [cos_theta_opp_half, sin_theta_opp_half * complex_exp(phi_opp)]])

    # # To test this function:
    # rand_onb = onb_from_direction(random.random() * pi, random.random() * 2 * pi)
    # print(np.linalg.norm(rand_onb[0]))  # Should give 1
    # print(np.inner(rand_onb[0], rand_onb[1].conj()))  # Should give 0


def random_onb(dim=2):
    kets = np.empty((0, dim))
    for i in range(dim):
        # Pick random ket orthogonal to the previous ones
        if len(kets) > 0:
            orth_vectors = scipy.linalg.null_space(np.array(kets).conj()).T  # transpose because null_space returns a matrix containing column vectors
        else:
            orth_vectors = np.identity(dim)  # list of standard basis elements
        assert dim - i == len(orth_vectors)
        # Pick random vector in C^(dim-i)
        random_vec = np.random.rand(dim - i) + 1j * np.random.rand(dim - i)
        # Use random_vec as components for orth_vectors to generate random element of span(orth_vectors)
        new_ket = np.array([random_vec]) @ orth_vectors
        kets = np.r_[kets, normalise_vec(new_ket)]
    return kets


def random_unitary(dim=2):
    return random_onb(dim).T


def random_real_onb():
    print("Avoid the use of random_real_onb(); it doesn't sample uniformly.")
    return onb_from_direction(random.random() * pi, 0)


## COMMON VECTORS AND ONBs
sqrt2 = sqrt(2)

# Some common states.
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
ket_plus = 1 / sqrt2 * np.array([1, 1])
ket_minus = 1 / sqrt2 * np.array([1, -1])
phi_plus = 1 / sqrt2 * np.array([1, 0, 0, 1])
phi_plus_un = np.array([1, 0, 0, 1])  # unnormalised
rho_ctb_plusphiplus = 1 / 2 * proj(kron(ket_plus, phi_plus_un))
ket_ctb_ghz = 1 / sqrt2 * np.array([1, 0, 0, 0, 0, 0, 0, 1])
rho_ctb_ghz = 1 / 2 * proj(np.array([1, 0, 0, 0, 0, 0, 0, 1]))
ket_tcb_0phi = np.einsum('i,jk->jik', ket0, phi_plus.reshape((2, 2))).reshape(8)
rho_tcb_0phi = 1 / 2 * proj(np.einsum('i,jk->jik', ket0, phi_plus_un.reshape((2, 2))).reshape(8))

# Some common ONBs.
z_onb = onb_from_direction(0)
x_onb = onb_from_direction(pi / 2)
diag1_onb = onb_from_direction(pi / 4)  # Checked with my notes [p76] ✓
diag2_onb = onb_from_direction(3 * pi / 4)  # Checked with my notes [p77] ✓
ket_diag = diag1_onb[0]  # Ket corresponding to spin in the z+x direction, 'in between' 0 and +.


# Some common instruments
def kraus_op_to_cj(lin_map):
    """
    :param lin_map: A linear map (i.e. dOut x dIn matrix) acting on kets (not on density matrices!). E.g. a projection |psi><psi| = proj(psi).
    :return: The CJ representation, i.e. (1xlin_map)phi+(1xlin_map*)
    """
    assert len(lin_map.shape) == 2
    dOut, dIn = lin_map.shape
    phi_plus_un_dIn = np.identity(dIn).reshape(dIn ** 2)
    id_kron_map = kron(np.identity(dIn), lin_map)
    return id_kron_map @ proj(phi_plus_un_dIn) @ id_kron_map.conj().T


def random_orth_proj(dim, rank):
    kets = random_onb(dim)[0:rank]
    # Make projection that projects onto the space spanned by kets
    return np.einsum('ki,kj->ij', kets, kets.conj())  # 'k' indexes the kets, i and j the kets' dimensions.


def nondestr_cj_to_destr_cj(nondestr_cj):
    """ Traces out the output system from a CJ state. Assumes that the passed CJ state represents a CP map that has output system dim equal to input system dim. """
    assert len(nondestr_cj.shape) == 2 and nondestr_cj.shape[0] == nondestr_cj.shape[1]
    assert sqrt(len(nondestr_cj)) == int(sqrt(len(nondestr_cj)))
    dim = int(sqrt(len(nondestr_cj)))
    # nondestr[i] is a tensor of the shape L ⊗ M with L, M matrices
    # Indices correspond to LoMo, LiMi
    # convert to Lo, Mo, Li, Mi:
    nondestr_tns = nondestr_cj.reshape((dim, dim, dim, dim))
    # Now trace out M (the output quantum system) to make the instrument destructive.
    return np.einsum('ijkj', nondestr_tns)


def instr_vn_nondestr(onb):
    # return np.array([kraus_op_to_cj(proj(ket)) for ket in onb])
    return instr_proj_mmt_nondestr(onb[0])


def instr_vn_destr(onb):
    # result = np.array([kraus_op_to_cj(np.array([ket.conj()])) for ket in onb])  # '[ket.conj()]' is the bra version of ket (conjugate and turned into row vector)
    # # I checked that this gives the same as tracing out the output system from instr_vn_nondestr:
    # dim = len(onb[0])
    # assert np.max(np.abs(result - np.einsum('aijkj', instr_vn_nondestr(onb).reshape((len(instr_vn_nondestr(onb)), dim, dim, dim, dim))))) < 1e-15
    return instr_proj_mmt_destr(onb[0])


def instr_measure_and_send_fixed_state(onb, fixed_ket):
    return np.array([kraus_op_to_cj(np.array([fixed_ket]).T @ [ket.conj()]) for ket in onb])  # |fixed_ket><ket|


def instr_proj_mmt_nondestr(ket):
    """ Returns instrument corresponding to the projective measurement with projections |ket><ket| and id - |ket><ket| """
    dim = len(ket)
    proj0 = proj(ket)
    proj1 = np.identity(dim) - proj0
    return np.array([kraus_op_to_cj(proj0), kraus_op_to_cj(proj1)])


def instr_weak_vN_nondestr(ket, noise, unitary=None):
    """ A nondestructive instrument that measures qubits with Kraus ops  `unitary * (sqrt(noise)*id/2 + i sqrt(1-noise)*|ket><ket|)`.
    So the unitary is applied after the noisy measurement. If unitary is None then np.identity(len(ket)) is used. """
    assert len(ket) == 2  # Not sure what to do with the noise terms if the dim is not 2. What to divide the identity by?
    if unitary is None:
        unitary = np.identity(2)
    proj0 = proj(ket)
    proj1 = np.identity(2) - proj0
    kraus0 = unitary @ (sqrt(noise/2) * np.identity(2) + 1j * sqrt(1 - noise) * proj0)
    kraus1 = unitary @ (sqrt(noise/2) * np.identity(2) + 1j * sqrt(1 - noise) * proj1)
    return np.array([kraus_op_to_cj(kraus0), kraus_op_to_cj(kraus1)])


def instr_proj_mmt_destr(ket):
    return np.array([nondestr_cj_to_destr_cj(cj) for cj in instr_proj_mmt_nondestr(ket)])


# instr_do_nothing = np.array([proj(phi_plus_un), np.zeros((4, 4), dtype='int')])     gives same as:
instr_do_nothing = np.array([kraus_op_to_cj(np.identity(2)).astype('int'), kraus_op_to_cj(np.zeros((2, 2))).astype('int')])  # outcome is 0 with probability 1


if __name__ == '__main__':
    # To test quantum_cor_from_complete_vn_mmts_new and make_pacb_xy_new:
    """rho_ctb, X1, X2, Y, c_onb = random_pure_density_matrix(True), \
                   [random_onb(), random_onb()], \
                   [random_onb(), random_onb()], \
                   [random_onb(), random_onb()], \
                   random_onb()
    assert np.all(quantum_cor_from_complete_vn_mmts_new(rho_ctb, X1, X2, Y, c_onb) == quantum_cor_nss_definitive(rho_ctb, X1, X2, Y, c_onb))
    instrs_A1 = [instr_vn_nondestr(i) for i in X1]
    instrs_A2 = [instr_vn_nondestr(i) for i in X2]
    instrs_B = [instr_vn_destr(i) for i in Y]
    instr_C = instr_vn_destr(c_onb)
    p_new = make_pacb_xy_new(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, 2, 2)
    p = make_pacb_xy(rho_ctb, X1, X2, Y, c_onb)
    print(np.all(p_new == p))  # Success!"""

    # TODO repeat this and see if same with new make_pacb_xy:
    # qm_cors = np.array(generate_some_quantum_cors_complete_vn_noTmmt('some_quantum_cors4_not_approximated.npy', num_of_random_cors=5000))
    # print('Done.')

    """rho_ctb = proj(np.array([1, 0, 0, 0, 0, 0, 0, 0]))
    cp_A1 = proj(phi_plus_un)
    cp_A2 = kron(proj(ket1), proj(ket0))
    cp_C = proj(ket0)
    tau_Ttilde = np.identity(2)
    cp_B = proj(ket0)
    taus = kron(rho_ctb.T, cp_A1.T, cp_A2.T, cp_C.T, tau_Ttilde, cp_B.T)

    phi_plus_dT = np.identity(2)
    phi_plus_dB = np.identity(2)
    # CO1 part of W. See [p72].  Tensor order before reordering of the Einstein indices: Bout*, B~in, Cout*, C~in, Tout*, A1in, A1out^*, A2in, A2out*, T~in
    # and after reordering: Cout*, Tout*, Bout*, A1in, A1out*, A2in, A2out*, C~in, T~in, B~in.
    W_CO1 = np.einsum('ij,k,l,mn,op,qr->kminopqlrj', phi_plus_dB, ket0, ket0, phi_plus_dT, phi_plus_dT, phi_plus_dT).reshape(1024)
    taus_test = kron(proj(ket1), np.identity(512))
    print(np.trace(np.matmul(proj(W_CO1), taus_test)))
    print(np.trace(np.matmul(proj(W_CO1), taus)))"""

    # TODO continue testing this:
    """cor = make_pacb_xy(rho_ctb=random_pure_density_matrix(True),
                       # instrs_A1=[instr_do_nothing, instr_measure_and_send_fixed_state(z_onb, ket0)],
                       # instrs_A2=[instr_do_nothing, instr_measure_and_send_fixed_state(z_onb, ket0)],
                       instrs_A1=[instr_measure_and_send_fixed_state(random_onb(), random_onb()[0]), instr_measure_and_send_fixed_state(random_onb(), random_onb()[0])],
                       instrs_A2=[instr_measure_and_send_fixed_state(random_onb(), random_onb()[0]), instr_measure_and_send_fixed_state(random_onb(), random_onb()[0])],
                       instr_C=instr_vn_destr(x_onb),
                       instrs_B=[instr_vn_destr(random_onb()), instr_vn_destr(random_onb())],
                       dT=2, dB=2) \
        .reshape(128)
    vector_space_utils.write_cor_to_file(cor, 'tmp')

    cor_nss_homog = vector_space_utils.full_acb_to_nss_homog(cor, 2 ** 18)
    print('Largest known LC violation:', towards_lc.maximum_violation_of_known_lc_facets(np.array([cor_nss_homog])))
    print('Largest Caus2 violation:', towards_lc.maximum_violation_of_caus2_facets(np.array([cor_nss_homog])))"""

    """
    instrs_A1, instrs_A2, instr_C, instrs_B = quantum_setup_two_channels()
    # sample random initial states
    for i in range(100):
        rho_ctb_random = random_pure_density_matrix(dim=16, allow_complex=True)
        cor = make_pacb_xy(rho_ctb_random, instrs_A1, instrs_A2, instr_C, instrs_B, 4, 2)
        cor_nss = vector_space_utils.full_acb_to_nss_homog(cor)
        if not lp_for_membership.lp_without_vertices_nss_coords(cor_nss, method='highs').success:
            print('\nFound violation!')
            break
        else:
            print('Checked %d random initial states' % i, end='\r')
            sys.stdout.flush()
    """

    # TODO fix this!
    cor_nss = quantum_cor_nss_from_complete_vn_mmts_noTmmt(rho_ctb_plusphiplus, [z_onb, x_onb], [z_onb, x_onb], z_onb, [z_onb, x_onb])
    cor_full = vs.construct_NSS_to_full_homogeneous() @ cor_nss
    cor_full = 1 / cor_full[-1] * cor_full[:-1]
    vs.write_cor_to_file(cor_full, 'tmp')
    swap_A1_A2_matrix = symmetry_utils.full_perm_to_symm(lambda a1, a2, c, b, x1, x2, y: (a2, a1, c, b, x2, x1, y))
    print(np.max(np.abs(cor_full - swap_A1_A2_matrix @ cor_full)))  # TODO WHY IS THIS NOT ZERO???????
    print(vs.is_in_NSCO1(cor_full, tol=1e-8))
    print(vs.is_in_NSCO2(cor_full, tol=1e-8))

    # qm_cors = generate_some_quantum_cors_complete_vn(num_of_random_cors=10)
