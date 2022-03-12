import cmath
import functools
import random
from math import pi, cos, sin, sqrt
import numpy as np
import itertools

import utils
import vector_space_utils

sqrt2 = sqrt(2)


## GENERAL UTILITY FUNCTIONS

def proj(vector):
    """ Returns the matrix representing projection onto the given vector.
     The vector should be a ROW vector, i.e. of shape (d,) """
    return np.einsum('i,j->ij', vector, vector.conj())


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
    :param perm: the permutaiton along which to shuffle the Kronecker factors, provided as a length-n list;
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
    :param perm: the permutaiton along which to shuffle the Kronecker factors, provided as a length-n list;
        e.g. (1,0,3,2) swaps the order of the first two and the last two factors.
    :return The permuted Kronecker product. Example: reshuffle(kron(A,B,C,D), (0,1,3,2)) gives the same as kron(B,A,D,C).
        It is crucial that all vectors A,B,C,D are assumed to be of length 2. But you can also use this function for
        power-of-2-length vectors: e.g. if A,B are two-dimensional and C is four-dimensional, then
        reshuffle(kron(A,B,C), (0,2,1)) gives the same as kron(A,C,B).

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


## COMMON VECTORS AND ONBs

# Some common vectors.
ket0 = np.array([1, 0])
ket1 = np.array([0, 1])
ket_plus = 1 / sqrt2 * np.array([1, 1])
ket_minus = 1 / sqrt2 * np.array([1, -1])
phi_plus = 1 / sqrt2 * np.array([1, 0, 0, 1])
phi_plus_un = np.array([1, 0, 0, 1])  # unnormalised
ket_werner = 1 / sqrt2 * np.array([1, 0, 0, 0, 0, 0, 0, 1])


# Some common ONBs
def onb_from_direction(theta, phi=0.):
    """ Returns the mmt ONB corresponding to measurement of a spin in the specified direction, using the Bloch representation.
     :param theta: polar angle of direction, in (0,π)
     :param phi: azimuthal angle of direction, in (0,2π)
     :return a tuple (v1, v2) of orthogonal, normalised 2d complex vectors.
    """
    if phi != 0:
        print("Code (not just onb_from_direction()) might not yet work with complex ONBs!")

    # The other vector will be given by the opposite direction in the Bloch sphere:
    theta_opp, phi_opp = pi - theta, phi + pi
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


z_onb = onb_from_direction(0)
x_onb = onb_from_direction(pi / 2)
diag1_onb = onb_from_direction(pi / 4)  # Checked with my notes [p76] ✓
diag2_onb = onb_from_direction(3 * pi / 4)  # Checked with my notes [p77] ✓
ket_diag = diag1_onb[0]  # Ket corresponding to spin in the z+x direction, 'in between' 0 and +.


## PROCESS OPERATORS
def process_operator_1switch_1mmt():
    """ Returns process operator of one switch approach, viz. the process operator on [p72] of notebook.
    The order of tensor factors is as in probability_one_switch_approach():
            (Cout*, Tout*, Bout*, A1in, A1out*, A2in, A2out*, C~in, T~in, B~in).
    """
    # Outline;
    #  - Construct CO1 part of W (this is a vector, not an operator)
    #  - Construct CO2 part of W
    #  - Make sure that the order of tensor factors in both parts are equal to the one promised above
    #  - Add the parts together to get W
    #  - return the matrix |W><W| = proj(W)
    # Work with row vectors, i.e. of shape (d,), because those work with our reshuffling function.

    # TODO maybe need to specify dtype="complex_" somewhere!

    # CO1 part of W. See [p72].  Tensor order: Bout*, B~in, Cout*, C~in, Tout*, A1in, A1out^*, A2in, A2out*, T~in
    W_CO1 = kron(phi_plus_un, ket0, ket0, phi_plus_un, phi_plus_un, phi_plus_un)

    # CO2 part of W. See [p72].  Tensor order: Bout*, B~in, Cout*, C~in, Tout*, A2in, A2out^*, A1in, A1out*, T~in
    W_CO2 = kron(phi_plus_un, ket1, ket1, phi_plus_un, phi_plus_un, phi_plus_un)

    # Correct the tensor orders
    W_CO1_ordered = reshuffle_kron_vector(W_CO1, (2, 4, 0, 5, 6, 7, 8, 3, 9, 1))
    W_CO2_ordered = reshuffle_kron_vector(W_CO2, (2, 4, 0, 7, 8, 5, 6, 3, 9, 1))

    # Add together and return projection (which is the process operator)
    W_total_ordered = W_CO1_ordered + W_CO2_ordered
    return proj(W_total_ordered)


def dependence_of_c_on_y_in_ptilde1(rho_ctb, X1, X2, Y, c_onb):
    """ Returns a measure of the dependence of c on y in the probability distribution p~1 defined in Thm 14/1/22 [p91].
    This is relevant to the one_switch_4mmts scenario, with no setting for the measurement on c.
    NOTE This function ONLY works properly if p(c = |0>)) is nonzero, i.e. if rho_ctb restricted to c is not |1>.
    """
    # Make p^1(ab|xy) correlation
    W_CO1 = kron(phi_plus_un, ket0, ket0, phi_plus_un, phi_plus_un, phi_plus_un)
    W_CO1_ordered = reshuffle_kron_vector(W_CO1, (2, 4, 0, 5, 6, 7, 8, 3, 9, 1))
    proc_op_1 = proj(W_CO1_ordered)
    cor_p1ab_xy = np.empty((2,) * 6)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=6):
        a_1, a_2, b, x_1, x_2, y = setting_outcome_tuple
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(
            ket0).T  # conditioning on c = ket0. This should be same as tau_Ctilde = identity, since we're using W_CO1. I checked it, is is true!
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        cor_p1ab_xy[setting_outcome_tuple] = np.trace(np.matmul(proc_op_1, taus))
    # Note that cor_p1ab_xy is not normalised (it has normalisation p(c=ket0), where p is generated by proc_op_total). We could normalise, but it isn't necessary.

    # Make p(abc|xy) correlation
    proc_op_total = process_operator_1switch_1mmt()
    cor_pabc_xy = np.empty((2,) * 7)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=7):
        a_1, a_2, b, c, x_1, x_2, y = setting_outcome_tuple
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(c_onb[c]).T
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        cor_pabc_xy[setting_outcome_tuple] = np.trace(np.matmul(proc_op_total, taus))

    # We need p~^1(abc|xy) = p1(ab|xy) p(c|xyab)
    # First make p(c|xyab) = p(abc|xy) / p(ab|xy)
    cor_pab_xy = np.einsum('ijklmno->ijkmno', cor_pabc_xy)  # p(ab|xy)
    # We now want 1 / p(ab|xy). Where p(ab|xy)=0, we must have p1(ab|xy)=0, because we assume p(c=|0>) != 0 [see p91].
    # This means that p~^1(abc|xy) = p1(ab|xy) p(c|xyab) = 0. Hence we can safely set 1 / p(ab|xy) = 0 for these cases.
    cor_pab_xy_inv = np.reciprocal(cor_pab_xy, out=np.zeros_like(cor_pab_xy), where=cor_pab_xy != 0)
    cor_pc_xyab = np.einsum('ijklmno,ijkmno->ijklmno', cor_pabc_xy, cor_pab_xy_inv)
    # Now make p~^1(c|xy) = sum_a sum_b p1(ab|xy) p(c|xyab).
    cor_ptilde1c_xy = np.einsum('ijkmno,ijklmno->lmno', cor_p1ab_xy, cor_pc_xyab)

    # Alternatively, tracing out c allows to check that a_1 and a_2 ARE independent of y. Yay!

    diff_y = cor_ptilde1c_xy[:, :, :, 0] - cor_ptilde1c_xy[:, :, :, 1]
    print(diff_y)
    return np.sum(np.abs(diff_y)), np.einsum('ijkmno,ijklmno->ijklmno', cor_p1ab_xy, cor_pc_xyab)


def dependence_of_b_on_x_in_phat1(rho_ctb, X1, X2, Y, c_onb):
    """ Returns a measure of the dependence of b on x_1, x_2 in the probability distribution p-hat-1 defined on [p92].
    This is relevant to the one_switch_4mmts scenario, with no setting for the measurement on c."""
    # Make p^1(ab|xy) correlation
    W_CO1 = kron(phi_plus_un, ket0, ket0, phi_plus_un, phi_plus_un, phi_plus_un)
    W_CO1_ordered = reshuffle_kron_vector(W_CO1, (2, 4, 0, 5, 6, 7, 8, 3, 9, 1))
    proc_op_1 = proj(W_CO1_ordered)
    correlation_p1 = np.empty((2,) * 6)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=6):
        a_1, a_2, b, x_1, x_2, y = setting_outcome_tuple
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(
            ket0).T  # conditioning on c = ket0. This should be same as tau_Ctilde = identity, since we're using W_CO1. I checked it, is is true!
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        correlation_p1[setting_outcome_tuple] = np.trace(np.matmul(proc_op_1, taus))
    # Note that correlation_p1 is not normalised (it has normalisation p(c=ket0), where p is generated by proc_op_total). We could normalise, but it isn't necessary.

    # Make p(abc|xy) correlation
    proc_op_total = process_operator_1switch_1mmt()
    correlation_pabc = np.empty((2,) * 7)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=7):
        a_1, a_2, b, c, x_1, x_2, y = setting_outcome_tuple
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(c_onb[c]).T
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        correlation_pabc[setting_outcome_tuple] = np.trace(np.matmul(proc_op_total, taus))

    # The new correlation will be phat1(abc|xy) = p1(a|x) p(c|xa) p(b|xyac)
    # Make p1(a|x)
    correlation_p1ax = np.einsum('ijkmno->ijmno', correlation_p1)[:, :, :, :, 0]  # Is equal to [:, :, :, :, 1] ✓
    # Make p(c|xa) = p(ac|x) / p(a|x)
    correlation_pacx = np.einsum('ijklmno->ijlmno', correlation_pabc)[:, :, :, :, 0]  # p(ac|x)
    correlation_pax = np.einsum('ijlmn->ijmn', correlation_pacx)  # p(a|x)
    correlation_pc_xa = np.einsum('ijlmn,ijmn->ijlmn', correlation_pacx, np.reciprocal(correlation_pax))
    # Make p(b|xyac) = p(abc|xy) / p(ac|x)
    correlation_pb_xyac = np.einsum('ijklmno,ijlmn->ijklmno', correlation_pabc, np.reciprocal(correlation_pacx))
    # Finally, make phat1(abc|xy) = p1(a|x) p(c|xa) p(b|xyac) AND trace out a,c
    correlation_phat1_b_xy = np.einsum('ijmn,ijlmn,ijklmno->kmno', correlation_p1ax, correlation_pc_xa,
                                       correlation_pb_xyac)

    diff_x_1 = correlation_phat1_b_xy[:, 0, :, :] - correlation_phat1_b_xy[:, 1, :, :]  # NOTE this is only the dependence on x_1!
    return np.sum(np.abs(diff_x_1))

    # Gives 0.228, 0.0283, etc. So indeed dependence!


def dependence_of_ac_on_y_in_phacek1(rho_ctb, X1, X2, Y, c_onb):
    """ Returns a measure of the dependence of a,c on y in the probability distribution pˇ1 defined on [p104]. (NOTE this function assumes p(c=0|by)=1/2 and indep of by)
    This is relevant to the one_switch_4mmts scenario, with no setting for the measurement on c.
    NOTE This function ONLY works properly if p(c = |0>)|yb) is nonzero AND INDEP OF y,b, e.g. if CTB = ket_plus tens phi_plus
    """
    # We need to make phacek1 := p(b|y) p^1(a|x,by) p(c|abxy)
    # First make p(abc|xy)
    pabc_xy = make_pabc_xy(rho_ctb, X1, X2, Y, c_onb)
    # Make p(b|y)
    pb_y = np.einsum('ijklmno->kmno', pabc_xy)[:, 0, 0, :]
    # Make p(c|abxy)
    pab_xy = np.einsum('ijklmno->ijkmno', pabc_xy)
    pab_xy_inv = np.reciprocal(pab_xy, out=np.zeros_like(pab_xy), where=pab_xy != 0)
    pc_abxy = np.einsum('ijklmno,ijkmno->ijklmno', pabc_xy, pab_xy_inv)
    # Make p^1(a|x,by)
    p1ab_xy = 2 * make_p1ab_xy_unnormalised(rho_ctb, X1, X2, Y)
    p1b_xy = np.einsum('ijkmno->kmno', p1ab_xy)
    p1b_xy_inv = np.reciprocal(p1b_xy, out=np.zeros_like(p1b_xy), where=p1b_xy != 0)
    p1a_bxy = np.einsum('ijkmno,kmno->ijkmno', p1ab_xy, p1b_xy_inv)
    # Make phacek1(abc|xy)
    phacek1abc_xy = np.einsum('ko,ijkmno,ijklmno->ijklmno', pb_y, p1a_bxy, pc_abxy)

    # Calculate measure of dependence of ac on y
    phacek1ac_xy = np.einsum('ijklmno->ijlmno', phacek1abc_xy)
    dep = np.sum(np.abs(phacek1ac_xy[:, :, :, :, :, 0] - phacek1ac_xy[:, :, :, :, :, 1]))

    return phacek1abc_xy, dep


def III(rho_ctb, X1, X2, Y, c_onb):
    """ See [p106]. """

    # Want to make q(abc|xy) := 1/2 * p1(ac|x)p(b|xya,c=1) + 1/2 p2(ac|x)p(b|xya,c=2)
    p1ab_xy = 2 * make_p1ab_xy_unnormalised(rho_ctb, X1, X2, Y)
    p2ab_xy = 2 * make_p2ab_xy_unnormalised(rho_ctb, X1, X2, Y)
    p1a_x = np.einsum('ijkmno->ijmno', p1ab_xy)[:, :, :, :, 0]
    p2a_x = np.einsum('ijkmno->ijmno', p2ab_xy)[:, :, :, :, 0]
    pabc_xy = make_pabc_xy(rho_ctb, X1, X2, Y, c_onb)
    pac_xy = np.einsum('ijklmno->ijlmno', pabc_xy)
    pac_x = pac_xy[:, :, :, :, :, 0]
    pa_x_inv = reciprocal_or_zero(np.einsum('ijlmn->ijmn', pac_x))
    pc_ax = np.einsum('ijlmn,ijmn->ijlmn', pac_x, pa_x_inv)
    p1ac_x = np.einsum('ijmn,ijlmn->ijlmn', p1a_x, pc_ax)
    p2ac_x = np.einsum('ijmn,ijlmn->ijlmn', p2a_x, pc_ax)

    # Now make p(b|xya,c=1) and p(b|xya,c=2)
    pabc_xy_Z = make_pabc_xy(rho_ctb, X1, X2, Y, z_onb)  # note the z_onb
    pac_xy_Z_inv = reciprocal_or_zero(np.einsum('ijklmno->ijlmno', pabc_xy_Z))
    pb_acxy_Z = np.einsum('ijklmno,ijlmno->ijklmno', pabc_xy_Z, pac_xy_Z_inv)
    pb_a1xy = pb_acxy_Z[:, :, :, 0, :, :, :]  # p(b|xya,c=1)
    pb_a2xy = pb_acxy_Z[:, :, :, 1, :, :, :]  # p(b|xya,c=2)

    # Finally:
    qabc_xy = 1 / 2 * (np.einsum('ijlmn,ijkmno->ijklmno', p1ac_x, pb_a1xy) + np.einsum('ijlmn,ijkmno->ijklmno', p2ac_x, pb_a2xy))

    # Now see if it's equal to pabc_xy
    print(np.sum(np.abs(pabc_xy - qabc_xy)))

    # Gives 4.25 (on rho_ctb=proj(kron(ket_plus, phi_plus)), X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[z_onb, x_onb], c_onb=x_onb)
    #  --> so no, q is not equal to p.


def make_pabc_xy(rho_ctb, X1, X2, Y, c_onb):
    proc_op_total = process_operator_1switch_1mmt()
    pabc_xy = np.empty((2,) * 7)
    for a_1, a_2, b, c, x_1, x_2, y in itertools.product((0, 1), repeat=7):
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(c_onb[c]).T
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        pabc_xy[a_1, a_2, b, c, x_1, x_2, y] = np.trace(np.matmul(proc_op_total, taus))
    return pabc_xy

def make_pacb_xy(rho_ctb, X1, X2, Y, c_onb):
    return make_pabc_xy(rho_ctb, X1, X2, Y, c_onb).swapaxes(2,3)


def make_pabc_xy_NSS_coords(rho_ctb, X1, X2, Y, c_onb):
    full_coords = make_pabc_xy(rho_ctb, X1, X2, Y, c_onb).reshape(2 ** 7)
    return vector_space_utils.construct_full_to_NSS_matrix(8, 2, 4, 2) @ full_coords


def print_pabc_xy_NSS_coords(rho_ctb, X1, X2, Y, c_onb):
    print(' '.join(map(str, make_pabc_xy_NSS_coords(rho_ctb, X1, X2, Y, c_onb))))


def make_p1ab_xy_unnormalised(rho_ctb, X1, X2, Y):
    W_CO1 = kron(phi_plus_un, ket0, ket0, phi_plus_un, phi_plus_un, phi_plus_un)
    W_CO1_ordered = reshuffle_kron_vector(W_CO1, (2, 4, 0, 5, 6, 7, 8, 3, 9, 1))
    proc_op_1 = proj(W_CO1_ordered)
    p1ab_xy = np.empty((2,) * 6)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=6):
        a_1, a_2, b, x_1, x_2, y = setting_outcome_tuple
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(
            ket0).T  # conditioning on c = ket0. This should be same as tau_Ctilde = identity, since we're using W_CO1. I checked it, is is true!
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        p1ab_xy[setting_outcome_tuple] = np.trace(np.matmul(proc_op_1, taus))
    return p1ab_xy


def make_p2ab_xy_unnormalised(rho_ctb, X1, X2, Y):
    W_CO2 = kron(phi_plus_un, ket1, ket1, phi_plus_un, phi_plus_un, phi_plus_un)
    W_CO2_ordered = reshuffle_kron_vector(W_CO2, (2, 4, 0, 7, 8, 5, 6, 3, 9, 1))
    proc_op_2 = proj(W_CO2_ordered)
    p2ab_xy = np.empty((2,) * 6)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=6):
        a_1, a_2, b, x_1, x_2, y = setting_outcome_tuple
        tau_ctb = rho_ctb.T
        tau_a_1 = kron(proj(X1[x_1][a_1]), proj(X1[x_1][a_1])).T
        tau_a_2 = kron(proj(X2[x_2][a_2]), proj(X2[x_2][a_2])).T
        tau_Btilde = proj(Y[y][b]).T
        tau_Ctilde = proj(
            ket0).T  # conditioning on c = ket0. This should be same as tau_Ctilde = identity, since we're using W_CO1. I checked it, is is true!
        tau_Ttilde = np.identity(2)
        taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
        p2ab_xy[setting_outcome_tuple] = np.trace(np.matmul(proc_op_2, taus))
    return p2ab_xy


def reciprocal_or_zero(array):
    return np.reciprocal(array, out=np.zeros_like(array), where=array != 0)


def quantum_cor_in_panda_format_nss(rho_ctb, X1, X2, Y, c_onb):
    cor = make_pacb_xy(rho_ctb, X1, X2, Y, c_onb).reshape((128,))
    cor_approx = utils.approximate(cor, [n/32. for n in range(32+1)])
    cor_approx_nss = vector_space_utils.construct_full_to_NSS_matrix(8,2,4,2) @ cor_approx
    from fractions import Fraction
    return ' '.join([str(Fraction(value)) for value in cor_approx_nss])


if __name__ == '__main__':
    ## Showing that p(c | x_1, x_2, y) can generally depend on y (i.e. 'disproving (iii)'):
    """
    tau_ctb = proj(kron(ket_plus, ket0, ket0))
    tau_a_1 = (proj(np.array([1, 0, 0, 0])) + proj(np.array([0, 0, 0, 1]))).T
    # tau_a_2 = (proj(np.array([1, 0, 0, 0])) + proj(np.array([0, 0, 0, 1]))).T
    tau_a_2 = (proj(1 / 2 * np.array([1, 1, 1, 1])) + proj(1 / 2 * np.array([1, -1, -1, 1]))).T
    # tau_a_1 = (proj(1 / 2 * np.array([1, 1, 1, 1])) + proj(1 / 2 * np.array([1, -1, -1, 1]))).T
    tau_Ctilde = proj(ket_plus).T
    tau_Ttilde = np.identity(2)
    tau_Btilde = np.identity(2)
    taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)
    process_op = process_operator_1switch_1mmt()
    print(np.trace(np.matmul(process_op, taus)))
    """

    ## Trying to see if p~^1(a_1, a_2, c | x_1, x_2, y) depends on y:

    """phacek1abc_xy, dep = dependence_of_ac_on_y_in_phacek1(
        rho_ctb=proj(kron(ket_plus, phi_plus)),
        # rho_ctb=proj(normalise_vec(np.random.rand(8))),
        X1=[z_onb, x_onb],  # X1 = [random_real_onb(), random_real_onb()] # TODO see below
        X2=[z_onb, x_onb],  # TODO change back to z_onb, x_onb when NaN problem resolved
        Y=[z_onb, x_onb],  # Y = [random_real_onb(), random_real_onb()]
        c_onb=x_onb)  # c_onb = x_onb
    print(dep)"""  # NOTE this code takes 10-11 seconds on my laptop, 13-14 seconds on google cloud c2 VM

    # III(rho_ctb=proj(kron(ket_plus, phi_plus)),
    #     X1=[z_onb, x_onb],
    #     X2=[z_onb, x_onb],
    #     Y=[z_onb, x_onb],
    #     c_onb=x_onb)

    # Gave 0.2972878186211555 for some random ONBs. So definitely dependence between y and c! (If code is correct)
    # For X1=Z,X, X2=diag1,2, Y=Z,X, c_onb=x_onb, and some random rho_ctb, I got 0.2535096933657037.

    # Using X1,X2,Y = random and c_onb = x_onb:
    # rho_ctb = ket0,ket0,ket0: gives 4e-16 ✓
    # rho_ctb = ket+,ket0,ket0: gives 3e-16 ✓
    # rho_ctb = phi+     ,ket0: gives 1.5e-16 ✓ all as expected
    # Now entangle c and b:
    # rho_ctb = 'ikj',phi_plus,ket0: gives 0.0804 and 0.0885 and 0.378.  ✓ NOTE Yay, gives dependence! So this is a candidate for LDCO violation---because the 'Method (I)' proof on [p91] doesn't work for it.
    # TODO left to check this for non-random X1,X2,Y. First solve div by 0 problem
    # Now entangle t and b:
    # rho_ctb = ket+,phi_+          : gives 0.224 and 0.0555 and 0.112 and 0.0997. NOTE Unexpected! But actually makes sense? Maybe?
    # Now try Werner state (should logically also work, following from fact that CB entangled already works, let alone TB entangled).
    # rho_ctb = werner              : gives 0.0347 and 0.283 ✓

    # NOTE See [p93] for the results with non-random X1,X2,Y.

    print('ready')
