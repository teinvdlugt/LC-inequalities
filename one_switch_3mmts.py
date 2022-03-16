# Setup:
# Alice has a switch and has two measurements (x_1, a_1), (x_2, a_2); Bob has just a mmt (b, y). The control is not measured.


import itertools
import numpy as np
import polytope_utils
import quantum_utils as qm
from quantum_utils import kron, proj, reshuffle_kron_vector, phi_plus, ket0, ket_diag, ket_plus, z_onb, x_onb, \
    diag1_onb, diag2_onb

one_switch_3mmts_dim = 64  # So this will be the length of the np vectors representing individual correlations/'behaviours'


def vertices_one_switch_3mmts():
    """ Returns np array of vertices on [p69] of notebook. TODO Does NOT yet use reduced dimension
    Dimensions are ordered as follows: p(000|000), p(000|001), p(000|010), p(000|011), p(000|100), etc., i.e. lexicographically,
    where the variables are ordered as p(a_1, a_2, b | x_1, x_2, y).
    """
    ## CO1
    # General form of CO1 vertices: p(a_1 | x_1) p(b | x_1, y) p(a_2 | x_1, x_2, y)  with all three p's deterministic
    # Should loop over deterministic functions f_a_1(x_1), f_b(x_1,y), f_a_2(x_1,x_2,y). Such a triple corresponds exactly to a vertex of the above form
    CO1_num_of_vertices = 2 ** (2 + 2 ** 2 + 2 ** 3)
    CO1_vertices = np.zeros((CO1_num_of_vertices,
                             one_switch_3mmts_dim))  # If encountering memory/speed problems, can look into if can make this dtype int32 or int16 instead of float.
    current_vertex = 0
    for f_a_1 in polytope_utils.list_of_fns_from_n_to_m(2, 2):
        for f_b in polytope_utils.list_of_fns_from_n_to_m(4, 2):
            for f_a_2 in polytope_utils.list_of_fns_from_n_to_m(8, 2):
                # the vertex is specified by f_a_1, f_b and f_a_2. Now feed it into our vertex list
                current_component = 0  # the index in the numpy array will be given by [current_vertex][current_component]
                for var_values in itertools.product((0, 1), repeat=6):
                    a_1 = var_values[0]
                    a_2 = var_values[1]
                    b = var_values[2]
                    x_1 = var_values[3]
                    x_2 = var_values[4]
                    y = var_values[5]
                    # Compute the value of p(a_1 a_2 b | x_1 x_2 y) for the above values of a_1, a_2, b, x_1, x_2, y
                    # Basically, it's 1 iff the values a_1 a_2 b correspond to the values computed by the functions f_a_1, etc.
                    # The value is already loaded as 0, so only change it when it should be 1.
                    if f_a_1[x_1] == a_1 and f_b[x_1 * 2 + y] == b and f_a_2[x_1 * 4 + x_2 * 2 + y] == a_2:
                        CO1_vertices[current_vertex][current_component] = 1

                    current_component += 1
                current_vertex += 1

    if CO1_num_of_vertices != current_vertex:
        print("Something went wrong")
        return None

    ## CO2  TODO better to make a general function where you just specify dependencies, to not repeat code
    # General form: p(a_2 | x_2) p(b | x_2, y) p(a_1 | x_1, x_2, y)
    CO2_num_of_vertices = 2 ** (2 + 2 ** 2 + 2 ** 3)
    CO2_vertices = np.zeros((CO2_num_of_vertices, one_switch_3mmts_dim))
    current_vertex = 0
    for f_a_2 in polytope_utils.list_of_fns_from_n_to_m(2, 2):
        for f_b in polytope_utils.list_of_fns_from_n_to_m(4, 2):
            for f_a_1 in polytope_utils.list_of_fns_from_n_to_m(8, 2):
                # the vertex is specified by f_a_2, f_b and f_a_1. Now feed it into our vertex list
                current_component = 0  # the index in the numpy array will be given by [current_vertex][current_component]
                for var_values in itertools.product((0, 1), repeat=6):
                    a_1 = var_values[0]  # This has to be exactly the same as in the CO1 case!
                    a_2 = var_values[1]
                    b = var_values[2]
                    x_1 = var_values[3]
                    x_2 = var_values[4]
                    y = var_values[5]
                    # Compute the value of p(a_1 a_2 b | x_1 x_2 y) for the above values of a_1, a_2, b, x_1, x_2, y
                    # Basically, it's 1 iff the values a_1 a_2 b correspond to the values computed by the functions f_a_2, etc.
                    # The value is already loaded as 0, so only change it when it should be 1.
                    if f_a_2[x_2] == a_2 and f_b[x_2 * 2 + y] == b and f_a_1[x_1 * 4 + x_2 * 2 + y] == a_1:
                        CO2_vertices[current_vertex][current_component] = 1

                    current_component += 1
                current_vertex += 1

    if CO2_num_of_vertices != current_vertex:
        print("Something went wrong")
        return None

    ## Merge CO1 and CO2
    all_vertices = np.r_[CO1_vertices, CO2_vertices]
    # Remove duplicate vertices. Those arise because CO1 and CO2 overlap. For larger problems, should probably do this in a cleverer way.
    all_vertices = np.unique(all_vertices, axis=0)
    print("There are " + str(len(all_vertices)) + " vertices in total.")  # 32704: agrees with calculated number! [p69]

    return all_vertices


def qm_corr_one_switch_3mmts(X1, X2, Y, rho_ctb, process_operator=None):
    """ Calculates the probability distribution p(a_1 a_2 b | x_1 x_2 y), where x_1 ranges over values in X1 and
    a_1 ranges over {0,1}. Hence:
    :param X1: should be a 2-tuple (X1[0], X1[1]), where X1[0] is a 'measurement setting', i.e.
        an ONB: X1[0] = (X1[0][0], X1[0][1]), with X1[0][0] a 2d vector and X1[0][1] an orthogonal 2d vector (shape (2,)). Outcome
        a_1 = 0 for setting x_1 corresponds to X1[x_1][0] being measured, while a_1 = 1 for that setting means X1[x_1][1] was measured.
        (In summary, X1 should have shape (2,2,2)).
    :param X2: sim.
    :param Y:  sim.
    :param rho_ctb: an 8x8 matrix representing the initial state fed into the one-switch configuration.
    :param process_operator: The process operator operator of the one switch approach, provide if you already have it.
        Defaults to None, in which it is constructed anew.
    :return a np vector of shape (one_switch_3mmts_dim,) representing the correlation_ptilde1. """

    # Construct the process operator once
    if process_operator is None:
        process_operator = qm.process_operator_1switch_1mmt()

    # The array that we'll fill & return:
    correlation = np.empty(one_switch_3mmts_dim)
    # Loop over measurement settings
    current_index = 0  # (We could alternatively first create a shape (2,2,2,2,2,2) correlation_ptilde1 tensor and flatten it afterwards)
    for setting_outcome_tuple in itertools.product((0, 1), repeat=6):
        a_1, a_2, b, x_1, x_2, y = setting_outcome_tuple
        correlation[current_index] = qm_prob_one_switch_3mmts(process_operator, X1[x_1][a_1], X2[x_2][a_2], Y[y][b],
                                                              rho_ctb)
        current_index += 1
    return correlation


def qm_prob_one_switch_3mmts(process_operator, a_1, a_2, b,
                             rho_ctb):  # TODO test this function on some simple cases
    """ process_op is assumed to be expressed in the ordered basis
            (Cout*, Tout*, Bout*, A1in, A1out*, A2in, A2out*, C~in, T~in, B~in)
    a_1, a_2, b should all be 2d ROW vectors (numpy shape (2,)). rho_ctb is an 8x8 matrix.
    Note that this function does not depend on the complete ONBs x_1 = (a_1, x_1[1]). That is because we assume
    the alternative outcome, i.e. the vector x_1[1], to be orthogonal to a_1, and given that fact, the probability
    of detecting a_1 does not depend on the exact orthogonal vector x_1[1]. (Because tau_a_1 defd below does not depend on x_1[1]).
    """
    # Reordering tensor factors seems not to be necessary in this case, because the above order (in which W is specified) is already convenient.
    # If I need to reorder tensor factors later for some reason, then see https://stackoverflow.com/questions/50883773/permute-string-of-kronecker-products
    # First compute the tensor product of all tau's, in the order specified above
    tau_ctb = rho_ctb.T
    tau_a_1 = kron(proj(a_1), proj(
        a_1)).T  # This is a matrix wrt basis (a_1, a_1^perp) for A_1^out and (a_1*, a_1^perp*) for A_1^in*, the DUAL BASIS to (a_1, a_1^perp).
    tau_a_2 = kron(proj(a_2), proj(a_2)).T  # TODO does that mean this code is correct? think about more
    tau_Ctilde = np.identity(2)  # Trace out the control. Transpose doesn't matter for identity
    tau_Ttilde = np.identity(2)  # Trace out the target.
    tau_Btilde = proj(b).T
    # Take tensor (Kronecker) product, in the right order (see above)
    taus = kron(tau_ctb, tau_a_1, tau_a_2, tau_Ctilde, tau_Ttilde, tau_Btilde)

    # Calculate probability using generalised Born rule
    return np.trace(np.matmul(process_operator, taus))


# To test qm_prob_one_switch_3mmts():
"""
process_op = process_operator_one_switch_approach()
print(probability_one_switch_approach(process_op, ket0, ket0, ket0, proj(kron(ket0, ket0, ket0))))  # Should give 1
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket0, proj(kron(ket0, ket0, ket0))))  # Should give 1/2
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket0, proj(kron(ket0, ket0, ket_plus))))  # Should give 1/4
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket_plus, proj(kron(ket0, ket0, ket_plus))))  # Should give 1/2
print(probability_one_switch_approach(process_op, ket_plus, ket_minus, ket_plus, proj(kron(ket0, ket0, ket_plus))))  # Should give 0
print(probability_one_switch_approach(process_op, ket_minus, ket0, ket0, proj(kron(ket1, ket0, ket0))))  # Should give 1/2
print(probability_one_switch_approach(process_op, ket_minus, ket0, ket0, proj(kron(ket1, ket0, ket0))))  # Should give 1/2
print(probability_one_switch_approach(process_op, ket_minus, ket0, ket0, proj(kron(ket0, ket0, ket0))))  # Should give 1/4
print(probability_one_switch_approach(process_op, ket_minus, ket0, ket0, proj(kron(ket_plus, ket0, ket0))))  # Should give (1/4+1/2)/2 = 3/8
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket0, proj(kron(ket0, phi_plus))))  # Should give 1/4
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket0, proj(kron(ket1, phi_plus))))  # Should give 1/8
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket0, proj(kron(ket_plus, phi_plus))))  # Should give (1/4+1/8)/2 = 3/16
print(probability_one_switch_approach(process_op, ket0, ket_plus, ket0, proj(kron(ket_minus, phi_plus))))  # Should give (1/4+1/8)/2 = 3/16
# All correct
"""

# Some possible initial states
ctb_cb_entangled_t_0 = proj(reshuffle_kron_vector(kron(phi_plus, ket0), (0, 2, 1)))  # density matrix
ctb_cb_entangled_t_diag = proj(reshuffle_kron_vector(kron(phi_plus, ket_diag), (0, 2, 1)))
ctb_tb_entangled_c_plus = proj(kron(phi_plus, ket_plus))
ctb_tb_entangled_c_diag = proj(kron(phi_plus, ket_diag))
ctb_ghz = 1 / 2 * proj(np.array([1, 0, 0, 0, 0, 0, 0, 1]))
# Can also use y vectors, but maybe/probably not necessary


## FINDING RESULTS

if __name__ == '__main__':
    # Generate vertices once
    one_switch_vertices = vertices_one_switch_3mmts()

    ## Testing for quantum violations
    print(polytope_utils.in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
        X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[diag1_onb, diag2_onb],
        rho_ctb=proj(kron(ket0, ket0, ket0)))))  # Should be True, ✓
    # # It gives warning 'A_eq is not of full row rank', but that's right because the sum of all vertices is a multiple of np.ones(64), which is the last row of A_eq.
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[diag1_onb, diag2_onb], rho_ctb=ctb_cb_entangled_t_0),
    #               tol=1e-14))  # Returns True
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[diag1_onb, diag2_onb], rho_ctb=ctb_cb_entangled_t_diag),
    #               tol=1e-14))  # Returns True
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[diag1_onb, diag2_onb], rho_ctb=ctb_ghz), tol=1e-14))  # Returns True
    # print(in_hull(one_switch_vertices, np.random.rand(64)))  # False, as expected
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[diag1_onb, diag2_onb], rho_ctb=ctb_ghz), tol=1e-14))  # Returns True
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[diag1_onb, diag2_onb], rho_ctb=ctb_tb_entangled_c_plus),
    #               tol=1e-14))  # Returns True (as expected)
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[z_onb, x_onb], Y=[z_onb, x_onb], rho_ctb=ctb_cb_entangled_t_0), tol=1e-14))  # Returns True
    # print(in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
    #     X1=[z_onb, x_onb], X2=[diag1_onb, diag2_onb], Y=[z_onb, x_onb], rho_ctb=ctb_cb_entangled_t_0),
    #               tol=1e-14))  # Returns True
    print(polytope_utils.in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(
        X1=[z_onb, x_onb], X2=[diag1_onb, diag2_onb], Y=[z_onb, x_onb], rho_ctb=ctb_ghz),
                                 tol=1e-13))  # Returns True but only until precision 1e-13

    ## Try a bunch of random configurations
    # process_op = qm.process_operator_1switch_1mmt()
    # for i in range(20):
    #     ctb_rand = np.random.rand(8)
    #     ctb_rand = proj(1 / np.linalg.norm(ctb_rand) * ctb_rand)
    #     X1_rand = [qm.random_onb(), qm.random_onb()]
    #     X2_rand = [qm.random_onb(), qm.random_onb()]
    #     Y_rand = [qm.random_onb(), qm.random_onb()]
    #     tol = 1e-8
    #     if not polytope_utils.in_hull(one_switch_vertices, qm_corr_one_switch_3mmts(X1_rand, X2_rand, Y_rand, ctb_rand, process_op), tol):
    #         print('Violated! With tolerance', tol)
    #         print('rho_ctb:', ctb_rand, 'X1:', X1_rand, '\nX2:', X2_rand, '\nY', Y_rand)
    #     print('Finished checking ' + str(i + 1) + ' random configs.')
