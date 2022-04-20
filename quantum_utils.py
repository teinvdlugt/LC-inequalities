import cmath
import functools
import random
from math import pi, cos, sin, sqrt
import numpy as np
import itertools

import towards_lc
import vector_space_utils


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

    phi_plus_dT = np.identity(dT)  # NOTE: unnormalised phi_+!
    phi_plus_dB = np.identity(dB)

    # CO1 part of W. See [p72].  Tensor order before reordering of the Einstein indices: Bout*, B~in, Cout*, C~in, Tout*, A1in, A1out^*, A2in, A2out*, T~in
    W_CO1 = np.einsum('ij,k,l,mn,op,qr->kminopqlrj', phi_plus_dB, ket0, ket0, phi_plus_dT, phi_plus_dT, phi_plus_dT).reshape(dB ** 2 * 2 * 2 * dT ** 6)

    # CO2 part of W. See [p72].  Tensor order before reordering of the Einstein indices: Bout*, B~in, Cout*, C~in, Tout*, A2in, A2out^*, A1in, A1out*, T~in
    W_CO2 = np.einsum('ij,k,l,mn,op,qr->kmipqnolrj', phi_plus_dB, ket1, ket1, phi_plus_dT, phi_plus_dT, phi_plus_dT).reshape(dB ** 2 * 2 * 2 * dT ** 6)
    # (Note: the only difference is that ket0 becomes ket1 for C and that A1 and A2, which correspond to no and pq, are interchanged.

    # Add together and return projection (which is the process operator)
    return proj(W_CO1 + W_CO2)


def make_pacb_xy(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, dT, dB):
    """ All provided instruments should be CJ reps of instruments in the form of 4x4 matrices, NOT transposes of those (so not 'taus'). """
    # Check if all arguments are valid instruments
    for instr in np.r_[instrs_A1, instrs_A2]:
        assert_is_quantum_instrument(instr, dT, dT)
    for instr in instrs_B:
        assert_is_quantum_instrument(instr, dB, 1)
    assert_is_quantum_instrument(instr_C, 2, 1)
    assert_is_quantum_instrument([rho_ctb], 1, 2 * dT * dB, num_of_outcomes=1)  # essentially checks that rho_ctb is a valid state (density matrix)

    # Make correlation
    proc_op_total = process_operator_switch()
    pacb_xy = np.zeros((2,) * 7)
    tau_Ttilde = np.identity(dT)  # trace out the output target system
    for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
        taus = kron(rho_ctb.T, instrs_A1[x1][a1].T, instrs_A2[x2][a2].T, instr_C[c].T, tau_Ttilde, instrs_B[y][b].T)
        born_prob = np.trace(np.matmul(proc_op_total, taus))
        if np.imag(born_prob) > 1e-15:
            print("WARNING - DETECTED A LARGE IMAGINARY VALUE IN PROBABILITY: p(%d,%d,b=%d,c=%d,%d,%d,%d) = %s" % (a1, a2, b, c, x1, x2, y, str(born_prob)))
        pacb_xy[a1, a2, c, b, x1, x2, y] = np.real(born_prob)
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


def quantum_cor_nss(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, dT=2, dB=2, common_multiple_of_denominators=None):
    cor_full = make_pacb_xy(rho_ctb, instrs_A1, instrs_A2, instr_C, instrs_B, dT, dB).reshape((128,))
    return vector_space_utils.full_acb_to_nss_homog(cor_full, common_multiple_of_denominators)


def quantum_cor_nss_from_complete_vn_mmts(rho_ctb, X1, X2, Y, c_onb, common_multiple_of_denominators=None):
    return quantum_cor_nss(rho_ctb=rho_ctb,
                           instrs_A1=[instr_vn_nondestr(onb) for onb in X1],
                           instrs_A2=[instr_vn_nondestr(onb) for onb in X2],
                           instr_C=instr_vn_destr(c_onb),
                           instrs_B=[instr_vn_destr(onb) for onb in Y],
                           dT=2, dB=2, common_multiple_of_denominators=common_multiple_of_denominators)


def generate_some_quantum_cors_complete_vn(file_to_save_to=None, num_of_random_cors=1000, common_multiple_of_denominators=None):
    """
    If common_multiple_of_denominators is not None, then the generated correlations are approximated by fractions with
    the given denominator.
    """
    qm_cors = [
        quantum_cor_nss_from_complete_vn_mmts(
            rho_ctb=rho_ctb_plusphiplus,  # CTB = |+> |phi+>
            X1=[z_onb, x_onb],
            X2=[z_onb, x_onb],
            Y=[z_onb, x_onb],
            c_onb=x_onb,
            common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts(
            rho_ctb=rho_tcb_0phi,
            X1=[z_onb, x_onb],
            X2=[z_onb, x_onb],
            Y=[diag1_onb, diag2_onb],
            c_onb=x_onb,
            common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts(
            rho_ctb=rho_tcb_0phi,
            X1=[z_onb, x_onb],
            X2=[z_onb, x_onb],
            Y=[z_onb, x_onb],
            c_onb=diag1_onb,
            common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts(
            rho_ctb=rho_tcb_0phi,
            X1=[diag1_onb, diag2_onb],
            X2=[z_onb, x_onb],
            Y=[z_onb, x_onb],
            c_onb=diag2_onb,
            common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts(
            rho_ctb=rho_ctb_ghz,  # CTB = |GHZ>
            X1=[diag1_onb, diag2_onb],
            X2=[diag1_onb, diag2_onb],
            Y=[diag1_onb, diag2_onb],
            c_onb=x_onb,
            common_multiple_of_denominators=common_multiple_of_denominators),
        quantum_cor_nss_from_complete_vn_mmts(
            rho_ctb=rho_ctb_ghz,  # CTB = |GHZ>
            X1=[z_onb, x_onb],
            X2=[z_onb, x_onb],
            Y=[z_onb, x_onb],
            c_onb=diag1_onb,
            common_multiple_of_denominators=common_multiple_of_denominators)]
    if file_to_save_to is not None:
        np.save(file_to_save_to, qm_cors)
    def random_quantum_setup():
        return random_3_qubit_pure_density_matrix(True), \
               [random_onb(), random_onb()], \
               [random_onb(), random_onb()], \
               [random_onb(), random_onb()], \
               random_onb()
    for i in range(0, num_of_random_cors):
        print("Generated %d / %d random qm cors" % (i, num_of_random_cors))
        qm_cors.append(quantum_cor_nss_from_complete_vn_mmts(*random_quantum_setup(), common_multiple_of_denominators=2 ** 17))
        if file_to_save_to is not None:
            np.save(file_to_save_to, qm_cors)
    return qm_cors


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


def random_3_qubit_pure_density_matrix(allow_complex=False):
    if allow_complex:
        random_ket = np.random.rand(8) + np.random.rand(8) * 1j
    else:
        random_ket = np.random.rand(8)
    return proj(normalise_vec(random_ket))


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


def random_onb():
    return onb_from_direction(random.random() * pi, random.random() * 2 * pi)


def random_real_onb():
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
rho_ctb_plusphiplus = 1 / 2 * proj(kron(ket0, phi_plus_un))
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
def instr_vn_nondestr(onb):
    num_of_outcomes = len(onb)
    instr = np.empty((num_of_outcomes, 4, 4), dtype=onb.dtype)  # a family of 4x4 matrices indexed by two binary indices
    for a in range(0, num_of_outcomes):
        instr[a] = kron(proj(onb[a]), proj(onb[a]))
    return instr


def instr_vn_destr(onb):
    num_of_outcomes = len(onb)
    instr = np.empty((num_of_outcomes, 2, 2), dtype=onb.dtype)  # a family of 4x4 matrices indexed by two binary indices
    for a in range(0, num_of_outcomes):
        instr[a] = proj(onb[a])
    return instr


def instr_measure_and_send_fixed_state(onb, fixed_ket):
    num_of_outcomes = len(onb)
    instr = np.empty((num_of_outcomes, 4, 4), dtype=onb.dtype)  # a family of 4x4 matrices indexed by two binary indices
    for a in range(0, num_of_outcomes):
        instr[a] = kron(proj(onb[a]), proj(fixed_ket))
    return instr


instr_do_nothing = np.array([proj(phi_plus_un), np.zeros((4, 4), dtype='int')])  # outcome is 0 with probability 1

if __name__ == '__main__':
    # To test quantum_cor_from_complete_vn_mmts_new and make_pacb_xy_new:
    """rho_ctb, X1, X2, Y, c_onb = random_3_qubit_pure_density_matrix(True), \
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
    qm_cors = np.array(generate_some_quantum_cors_complete_vn('some_quantum_cors4_not_approximated.npy', num_of_random_cors=5000))
    print('Done.')

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
    cor = make_pacb_xy(rho_ctb=random_3_qubit_pure_density_matrix(True),
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
    print('Largest Caus2 violation:', towards_lc.maximum_violation_of_caus2_facets(np.array([cor_nss_homog])))
