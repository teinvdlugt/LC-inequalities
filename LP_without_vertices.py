import itertools

import numpy as np
from scipy.optimize import NonlinearConstraint, LinearConstraint
import scipy.optimize
import time
import one_switch_4mmts
import quantum_utils as qm
import vector_space_utils as vs

## IN THIS SCRIPT WE ORDER THE VARIABLES AS p(a1 a2 c b | x1 x2 y), i.e. c and b INTERCHANGED!
# It is crucial to stick to this convention, e.g. when switching between full and NSS reps

# The LC constraints are:
#         (i)   0 = λ p^CO1 + (1-λ) p^CO2 - p
#         (ii)  p^CO1 ≥ 0  expressed in NSS coords
#         (iii) p^CO2 ≥ 0  expressed in NSS coords
#         (iv)  0 ≤ λ ≤ 1
#         (v)   sum_{a2 c} p^CO1(a1=0 a2 c | x1 0) - sum_{a2 c} p^CO1(a1=0 a2 c | x1 1) = 0  for all x1  (NS for a1=1 follows from this and prev constraints)
#         (vi)  sum_{a1 c} p^CO2(a1 a2=0 c | 0 x2) - sum_{a1 c} p^CO2(a1 a2=0 c | 1 x2) = 0  for all x2


def construct_constraint_i(correlation):
    # We need the function conv_sum(x) := λ p^CO1 + (1-λ) p^CO2, its Jacobian, and its Hessians.
    def conv_sum(x):  # return shape (86,)
        return np.array([x[0] * x[i + 1] + (1 - x[0]) * x[i + 87] for i in range(0, 86)])  # vector of length 86
    def conv_sum_J(x):  # returns shape (86, 173)
        # ith row in Jacobian is gradient of ith component of conv_sum. Jacobian has shape (86, 173)
        # first column contains lambda-derivatives of all components of conv_sum
        column1 = np.array([[x[i + 1] - x[i + 87], ] for i in range(0, 86)])
        # The next 86 columns contain just lambdas at every component, and next 86 have 1-lambda.
        columns2 = np.empty((86, 86))
        columns2.fill(x[0])
        columns3 = np.empty((86, 86))
        columns3.fill(1 - x[0])
        # Concatenate the rows to get the Jacobian
        return np.concatenate((column1, columns2, columns3), axis=1)
    def conv_sum_H(x, v):  # returns shape (173, 173)
        # x has length 173, v has length 86
        """ long version:
        result = np.zeros((86, 86))
        for i in range(0, 86):
            # The current constraint is x[0] * x[i + 1] + (1 - x[0]) * x[i + 87]. Want the Hessian of this and multiply it with v[i].
            # The only non-zero second order derivatives of this constraint are those wrt lambda(=x[0]) and (x[i+1] or x[i+87])
            # So only first row and first column have non-zero elements
            result[0, i + 1] += v[i]  # dλ d(i+1)
            result[0, i + 87] -= v[i]  # dλ d(i+87)
            result[i + 1, 0] += v[i]  # d(i+1) dλ
            result[i + 87, 0] -= v[i]  # d(i+87) dλ
        """
        # This is the first row, but also the first column (Hessians are symmetric):
        row0 = np.r_[[0], [v[i] for i in range(0, 86)], [-v[i] for i in range(0, 86)]]  # shape (86,)
        hessian = np.zeros((173, 173))
        # populate first row
        hessian[0] = row0
        # transpose and populate first row again
        hessian = hessian.T
        hessian[0] = row0
        return hessian
    p_NSS = full_to_NSS_rep(correlation)  # the same correlation but in NSS representation, i.e. shape (86,) vector
    return conv_sum, conv_sum_J, conv_sum_H, p_NSS


def construct_constraints_v_vi():
    # Constraint (v) is just two scalar constraints. The first constraint corresponds to a 173-length row in the (2,173) matrix.
    # It has 1s in positions corresponding to p^CO1(0 a2 c | 0 0) for each a2,c, and -1s at p^CO1(0 a2 c | 0 1)
    cons_v_matrix = np.zeros((2, 173))
    for a2, c, x1, x2 in itertools.product((0, 1), repeat=4):
        cons_v_matrix[x1, 1 + vs.concatenate_bits(0, a2, c, x1, x2)] = (-1) ** x2
    cons_vi_matrix = np.zeros((2, 173))
    for a1, c, x1, x2 in itertools.product((0, 1), repeat=4):
        cons_vi_matrix[x2, 87 + vs.concatenate_bits(a1, 0, c, x1, x2)] = (-1) ** x1
    cons_v_vi_matrix = np.r_[cons_v_matrix, cons_vi_matrix]
    cons_v_vi_lb = cons_v_vi_ub = np.zeros(4)
    return cons_v_vi_matrix, cons_v_vi_lb, cons_v_vi_ub


def in_LC_scipy_constr_minimisation(correlation, initial_guess=np.zeros(173)):
    """
    Returns true if there is a decomposition of correlation = λ p^CO1 + (1-λ) p^CO2 where p^CO1,2 is in NS^CO1,2.
    NOTE this is about LC, not LC*. (This relates to the formulation of constraints (v) and (vi) below.)
    :param correlation: numpy array of size (2,2,2,2,2,2,2) representing correlation p(a_1 a_2 b c | x_1 x_2 y).
    :param tol: tolerance value. Is passed on to scipy. Default value for scipy is 1e-8.
    scipy.optimize.linprog docs: https://docs.scipy.org/doc/scipy/reference/optimize.linprog-interior-point.html
    scipy.optimize.minimize docs: https://docs.scipy.org/doc/scipy/tutorial/optimize.html#id35
    """

    # Want to know if correlation = λ p^CO1 + (1-λ) p^CO2 where p^CO1,2 is in NS^CO1,2.
    # The 'unknowns' are lambda (a number), the 86 NSs coords of p^CO1, and the 86 NSS coords of p^CO2. That's 173 numbers in total.
    # These will all sit in one (173,) vector called x.
    # The LP problem is
    #   max.  0
    #   s.t.  (i)   p = λ p^CO1 + (1-λ) p^CO2
    #         (ii)  p^CO1 ≥ 0  expressed in NSS coords
    #         (iii) p^CO2 ≥ 0  expressed in NSS coords
    #         (iv)  0 ≤ λ ≤ 1
    #         (v)   sum_{a2 c} p^CO1(a1=0 a2 c | x1 0) - sum_{a2 c} p^CO1(a1=0 a2 c | x1 1) = 0  for all x1  (NS for a1=1 follows from this and prev constraints)
    #         (vi)  sum_{a1 c} p^CO2(a1 a2=0 c | 0 x2) - sum_{a1 c} p^CO2(a1 a2=0 c | 1 x2) = 0  for all x2

    # I use scipy's Trust-Region Constrained Algorithm https://docs.scipy.org/doc/scipy/tutorial/optimize.html#id35

    ## Constraint (i): is nonlinear (https://docs.scipy.org/doc/scipy/tutorial/optimize.html#defining-nonlinear-constraints)
    # We need the function conv_sum(x) := λ p^CO1 + (1-λ) p^CO2, its Jacobian, and its Hessians.
    conv_sum, conv_sum_J, conv_sum_H, p_NSS = construct_constraint_i(correlation)
    constraint_i = NonlinearConstraint(conv_sum, p_NSS, p_NSS, jac=conv_sum_J, hess=conv_sum_H)

    ## Constraint (ii)-(iv): are linear (https://docs.scipy.org/doc/scipy/tutorial/optimize.html#defining-linear-constraints)
    # Using NSS_to_full_matrix_but_weird @ x[1:87], we can get the full-dimension components of p^CO1, and demand that those are >= 0.
    # Except for the p(1111|x1x2y) (i.e. the last 8 entries); instead of those,  NSS_to_full_matrix_but_weird @ x[1:87]  contains
    # the entries p(1111|x1x2y) - 1. Hence those should be demanded instead to be >= -1.
    cons_ii_matrix = get_NSS_to_full_matrix_but_weird()  # shape (128,86)
    cons_ii_lb = np.r_[np.zeros(120), -np.ones(8)]  # shape (128,)
    # But this is wrong: the matrix acts on (86,)-vectors, not (173,)-vectors. Hence, pad the matrix with zeros.
    # But then we might just as well add constraint (iii) and (iv) in the same matrix. The resulting matrix will have one 1 in the top-left-hand
    # corner (for constraint 0 ≤ λ ≤ 1), then a block in the (almost) upper left hand corner for p^CO1>=0, and a block in the lower right hand
    # corner for p^CO2>=0. The big matrix has shape (257,173), since there are 257 scalar constraints (ii, iii, iv combined).
    cons_ii_iii_iv_matrix = np.r_[[np.concatenate(([1], np.zeros(172)), axis=0)],  # \                           shape (1,  173)  |
                                  np.concatenate(([[0]] * 128, cons_ii_matrix, np.zeros((128, 86))), axis=1),  # shape (128,173)  |- (257,173) together
                                  np.concatenate(([[0]] * 128, np.zeros((128, 86)), cons_ii_matrix), axis=1)]  # shape (128,173)  |
    cons_ii_iii_iv_lb = np.r_[[0], cons_ii_lb, cons_ii_lb]
    cons_ii_iii_iv_ub = np.r_[[1], [np.inf] * 256]
    constraint_ii_iii_iv = LinearConstraint(cons_ii_iii_iv_matrix, cons_ii_iii_iv_lb, cons_ii_iii_iv_ub)

    ## Constraint (v)-(vi): are linear.
    cons_v_vi_matrix, cons_v_vi_lb, cons_v_vi_ub = construct_constraints_v_vi()
    constraint_v_vi = LinearConstraint(cons_v_vi_matrix, cons_v_vi_lb, cons_v_vi_ub)

    ## Run algorithm!
    print('Running optimization algorithm...')
    result = scipy.optimize.minimize(fun=lambda x: 0,  # function to minimise
                                     x0=initial_guess,  # initial guess
                                     method='trust-constr',
                                     jac=lambda x: np.zeros(173),  # Jacobian of the minimisation function, which is R^173 -> R --- so simply the gradient
                                     hess=lambda x: np.zeros((173, 173)),  # Hessian of minimisation function
                                     constraints=[constraint_i, constraint_ii_iii_iv, constraint_v_vi],
                                     options={'verbose': 1})
    print('Message from scipy.optimize.minimize:', result.message)
    print('In LC!' if result.success else 'Not in LC!')
    return result


def in_LC_scipy_root(correlation, initial_guess=np.zeros(173)):
    """
    Uses multivariate root-finding algorithm to determine if correlation is in LC.
    :param correlation: A shape-(128,) vector representing a correlation in the full-dimensional representation.
    :param initial_guess: A shape-(173,) vector representing the initial guess (λ, p^CO1-NSS, p^CO2-NSS).
    """
    # Recall the constraints that we want to solve:
    #         (i)   p = λ p^CO1 + (1-λ) p^CO2
    #         (ii)  p^CO1 ≥ 0  expressed in NSS coords
    #         (iii) p^CO2 ≥ 0  expressed in NSS coords
    #         (iv)  0 ≤ λ ≤ 1
    #         (v)   sum_{a2 c} p^CO1(a1=0 a2 c | x1 0) - sum_{a2 c} p^CO1(a1=0 a2 c | x1 1) = 0  for all x1  (NS for a1=1 follows from this and prev constraints)
    #         (vi)  sum_{a1 c} p^CO2(a1 a2=0 c | 0 x2) - sum_{a1 c} p^CO2(a1 a2=0 c | 1 x2) = 0  for all x2
    # The equality constraints (i,v,vi) can be easily encoded as root constraints.
    # For the inequality constraints (ii—iv), note that p^CO1 ≥ 0 is satisfied for p^CO1 iff  np.min(p^CO1, 0)  has a root in p^CO1.

    # In the below, the functions called 'fun_...' are the ones that we have to find a root of.

    ## The equality constraints (i,v,vi)
    conv_sum, jac_i, _, p_NSS = construct_constraint_i(correlation)
    fun_i = lambda x: conv_sum(x) - p_NSS  # 86 scalar constraints (=> output shape is (86,)
    cons_v_vi_matrix, _, _ = construct_constraints_v_vi()
    fun_v_vi = lambda x: cons_v_vi_matrix @ x  # 4 scalar constraints
    jac_v_vi = cons_v_vi_matrix  # derivative of multiplication with a matrix is the matrix itself.

    ## The inequality constraints (ii—iv). Need to find root of e.g.
    # Constraint (ii): p^CO1 ≥ 0
    fun_ii = lambda x: np.minimum(NSS_to_full_rep(x[1:87]), 0)  # 128 scalar constraints
    deriv_of_NSS_to_full = get_NSS_to_full_matrix_but_weird()
    deriv_of_pCO1_NSS_to_full = np.concatenate(([[0, ], ] * 128, deriv_of_NSS_to_full, np.zeros((128, 86))), axis=1)
    def jac_ii(x):
        # Would like to do  jac_ii = lambda x: np.where(NSS_to_full_rep(x[1:87]) < 0, deriv_of_pCO1_NSS_to_full, 0)
        # but numpy doesn't like that the shapes don't match
        result = np.zeros((128, 173))
        for i in np.argwhere(NSS_to_full_rep(x[1:87]) < 0):
            result[i] = deriv_of_pCO1_NSS_to_full[i]
        return result
    fun_iii = lambda x: np.minimum(NSS_to_full_rep(x[87:173]), 0)  # 128 scalar constraints
    deriv_of_pCO2_NSS_to_full = np.concatenate(([[0, ], ] * 128, np.zeros((128, 86)), deriv_of_NSS_to_full), axis=1)
    def jac_iii(x):
        result = np.zeros((128, 173))
        for i in np.argwhere(NSS_to_full_rep(x[87:173]) < 0):
            result[i] = deriv_of_pCO2_NSS_to_full[i]
        return result
    fun_iv = lambda x: [min(x[0], 0), min(1 - x[0], 0)]  # 2 scalar constraints
    jac_iv = lambda x: np.concatenate(([[int(x[0] < 0)], [-int(x[0] > 1)]], np.zeros((2, 172))), axis=1)

    ## Combine all constraints into one function to find root of.
    # Combined, there are 86+4+128+128+2=348 constraints, so fun should be R^173 -> R^348
    fun = lambda x: np.r_[fun_i(x), fun_ii(x), fun_iii(x), fun_iv(x), fun_v_vi(x)]
    jac = lambda x: np.r_[jac_i(x), jac_ii(x), jac_iii(x), jac_iv(x), jac_v_vi]

    print('max(fun(initial_guess)):', max(fun(initial_guess)))

    # Solve!
    print('Running root finding algorithm...')
    start_time = time.time()
    sol = scipy.optimize.root(fun, method='lm',  # only method 'lm' seems to work when input and output of fun don't have same shapes
                              x0=initial_guess,
                              jac=jac, )
    # options={'xtol': 1e-14, 'ftol': 1e-14})
    # sol = scipy.optimize.minimize(lambda x: np.linalg.norm(fun(x)),
    #                               x0=initial_guess,
    #                               method='CG',
    #                               options={'disp': True},
    #                               tol=1e-10)
    print('That took %.2f seconds' % (time.time() - start_time))
    print('Reason for terminating:', sol.message)
    if sol.success:
        print('Successfulness of lm algorithm doesn\'t mean much')
    else:
        print('Either something went wrong or the correlation is not in LC.')
    return sol


def full_to_NSS_rep(cor, swap_b_and_c=False):
    # H  # TODO test this fn
    """
    NOTE this assumes that the order of the variables in `cor` is p(a1 a2 c b | x1 x2 y).
    :param cor: np array of shape (128,)
    :param swap_b_and_c: whether to swap position of b and c first. Correlations returned by
            one_switch_4mmts.qm_corr_one_switch_4mmts_3stngs() have b and c interchanged so this argument should be set to True.
    :return: np array of shape (86,)  (note dim(NSS) = 86)
    """
    if swap_b_and_c:
        cor = np.reshape((2,) * 7).swapaxes(2, 3).reshape((128,))
    return get_full_to_NSS_matrix() @ cor


full_to_NSS_matrix = None


def get_full_to_NSS_matrix():
    """
    Make and return matrix that converts shape (2^7,) vectors (full dimension) to shape (86,) vectors (NSS representation)
    ith row of the matrix represents ith NSS coordinate
    Each row is 128 long, and there are 86 rows.
    """
    global full_to_NSS_matrix
    if full_to_NSS_matrix is not None:
        return full_to_NSS_matrix
    else:
        print("Constructing full_to_NSS_matrix...")

    full_to_NSS_matrix = np.zeros((86, 128))
    current_row = 0  # keep track of current row in matrix that I'm filling

    # Range of the variables in the NSS representation. (See NSS representation)
    a1a2c_range = list(itertools.product((0, 1), repeat=3))[0:7]  # {0,1}^3 \ {(1,1,1)}
    b_range = (0,)
    x1_range = x2_range = y_range = (0, 1)

    # First the p(a1 a2 c | x1 x2) coords, for (a1,a2,c) ∊ a1a2c_range
    for (a1, a2, c), x1, x2 in itertools.product(a1a2c_range, x1_range, x2_range):
        # p(a1 a2 c | x1 x2) = sum_b p(a1 a2 c b | x1 x2 y=0)
        for b in (0, 1):
            full_to_NSS_matrix[current_row][vs.concatenate_bits(a1, a2, c, b, x1, x2, 0)] = 1
        current_row += 1

    # Secondly, the p(b | y) coords, for (b,y) ∊ b_range × y_range
    for b, y in itertools.product(b_range, y_range):
        # p(b | y) = sum_{a_1 a_2 c} p(a1 a2 c b | x1=0 x2=0 y)
        for a1, a2, c in itertools.product((0, 1), repeat=3):
            full_to_NSS_matrix[current_row][vs.concatenate_bits(a1, a2, c, b, 0, 0, y)] = 1
        current_row += 1

    # Finally, the p(a1 a2 c b | x1 x2 y) coords, again variables restricted to their appropriate ranges
    for (a1, a2, c), b, x1, x2, y in itertools.product(a1a2c_range, b_range, x1_range, x2_range, y_range):  # (sic. for the order of the vars)
        full_to_NSS_matrix[current_row][vs.concatenate_bits(a1, a2, c, b, x1, x2, y)] = 1
        current_row += 1

    return full_to_NSS_matrix


NSS_to_full_matrix_but_weird = None


def get_NSS_to_full_matrix_but_weird():
    """
    :return A shape-(128,86) matrix that converts NSS rep vectors (length 86) to full-dim rep vectors (length 128).
    NOTE: HOWEVER, the vector that you get by multiplying by this matrix does NOT contain the entries p(1,1,1,1|x1,x2,y);
    NOTE  in those indices instead contains the entries p(1,1,1,1|x1,x2,y) - 1. (Since note that you cannot get p(1,1,1,1|x1,x2,y)
    NOTE  from that just by matrix multiplication.)
    """
    global NSS_to_full_matrix_but_weird
    if NSS_to_full_matrix_but_weird is not None:
        return NSS_to_full_matrix_but_weird
    else:
        print("Constructing NSS_to_full_matrix...")

    """ NSS_to_full_matrix_but_weird = np.zeros((128, 86))
        NSS_II_offset = 28  # the index at which NSS coords of type II [see p113] start within a NSS vector
        NSS_III_offset = NSS_II_offset + 2  # the index at which NSS coords of type III [see p113] start within a NSS vector
        current_row = 0
        # Each row represents a tuple a1,a2,c,b,x1,x2,y
        for a1, a2, c, b, x1, x2, y in itertools.product((0, 1), repeat=7):
            # See [p113] in Sketchbook for expressions in terms of NSS coords
            if (a1, a2, c) != (1, 1, 1) and b != 1:
                # p(a1a2bc|x1x2y) appears in NSS itself
                # in this row, we just want a 1 in position NSS_III_offset + the index corresponding to p(a1 a2 c b | x1 x2 y) in NSS-III
                # since always b=0 in the NSS-III coords, we must leave b out from the concatenate_bits
                NSS_to_full_matrix_but_weird[current_row][NSS_III_offset + concatenate_bits(a1, a2, c, x1, x2, y)] = 1
            elif (a1, a2, c) == (1, 1, 1) and b != 1:
                # p(a1a2bc|x1x2y) = p(b|y) - sum_{ac != (1,1,1)} p(a1a2cb|x1x2y)
                # The index of p(b|y) is NSS_II_offset + y  (since only takes value 0 in NSS coords)
                NSS_to_full_matrix_but_weird[current_row][NSS_II_offset + y] = 1
                # The index of p(a1a2cb|x1x2y) is NSS_III_offset + concatenate_bits(a1, a2, c, x1, x2, y) --- note that again, we must leave b out
                for _a1, _a2, _c in list(itertools.product((0, 1), repeat=3))[0:7]:
                    NSS_to_full_matrix_but_weird[current_row][NSS_III_offset + concatenate_bits(_a1, _a2, _c, x1, x2, y)] = -1
            elif (a1, a2, c) != (1, 1, 1) and b == 1:
                # p(a1a2cb|x1x2y) = p(a1a2c|x1x2) - p(a1a2c b=0 |x1x2y)
                # The index of p(a1a2c|x1x2) is concatenate_bits(a1, a2, c, x1, x2)
                NSS_to_full_matrix_but_weird[current_row][concatenate_bits(a1, a2, c, x1, x2)] = 1
                # The index of p(a1a2c b=0 |x1x2y) is NSS_III_offset + concatenate_bits(a1, a2, c, x1, x2, y)
                NSS_to_full_matrix_but_weird[current_row][NSS_III_offset + concatenate_bits(a1, a2, c, x1, x2, y)] = -1
            elif (a1, a2, c) == (1, 1, 1) and b == 1:
                # (For the current values of x1,x2,y), the rows corresponding to p(a1a2cb|x1x2y) with a1a2cb != (1,1,1,1) have already been filled.
                # So can use those to calculate the current row, which is -sum_{(a1,a2,c,b)!=(1,1,1,1)} p(a1a2cb|x1x2y)
                # NOTE that this isn't p(1,1,1,1|x1x2y) but instead it is p(1,1,1,1|x1x2y) - 1.
                # The index of p(a1a2cb|x1x2y) in the previous rows is concatenate_bits(a1, a2, c, b, x1, x2, y)
                for _a1, _a2, _c, _b in list(itertools.product((0, 1), repeat=4))[0:15]:  # [0:15] because != (1,1,1,1)
                    NSS_to_full_matrix_but_weird[current_row] -= NSS_to_full_matrix_but_weird[concatenate_bits(_a1, _a2, _c, _b, x1, x2, y)]
            else:
                print("something went wrong")
            current_row += 1

        return NSS_to_full_matrix_but_weird"""

    return vs.construct_NSS_to_full_matrix_but_weird(8, 2, 4, 2)


"""
# To test full_to_NSS_rep() and NSS_to_full_rep(): see if their composition is the identity
qm_cor = one_switch_4mmts.qm_corr_one_switch_4mmts_3stngs(
    rho_ctb=qm.proj(qm.kron(qm.ket_plus, qm.phi_plus)),
    X1=[qm.random_real_onb(), qm.random_real_onb()],
    X2=[qm.random_real_onb(), qm.random_real_onb()],
    Y=[qm.random_real_onb(), qm.random_real_onb()],
    c_onb=qm.random_real_onb()).reshape((2,) * 7).swapaxes(2, 3).reshape((128,))
print('Constructed qm_cor')
qm_cor_NSS = full_to_NSS_rep(qm_cor)
print('Constructed qm_cor_NSS')
qm_cor_full_again = NSS_to_full_rep(qm_cor_NSS)
print('Constructed qm_cor_full_again')
print(np.all(almost_equal(qm_cor, qm_cor_full_again)))
print(np.sum(np.abs(qm_cor - qm_cor_full_again)))
_qm_cor = qm_cor.reshape((2,) * 7)
_qm_cor_full_again = qm_cor_full_again.reshape((2,) * 7)
"""


def NSS_to_full_rep(NSS_correlation):
    """ Converts (86,) vector into (128,) vector by multiplying with NSS_to_full_matrix_but_weird AND correcting for the weirdness. """
    full_rep_but_weird = get_NSS_to_full_matrix_but_weird() @ NSS_correlation
    # In place of p(1,1,1,1|x1x2y), full_rep_but_weird contains p(1,1,1,1|x1x2y) - 1. So add ones in those locations.
    for x1x2y in range(0, 8):
        full_rep_but_weird[0b1111000 + x1x2y] += 1
    return full_rep_but_weird

if __name__ == '__main__':
    qm_cor = one_switch_4mmts.qm_corr_one_switch_4mmts_3stngs(
        rho_ctb=qm.proj(qm.kron(qm.ket0, qm.ket0, qm.ket0)),
        X1=[qm.random_real_onb(), qm.random_real_onb()],
        X2=[qm.random_real_onb(), qm.random_real_onb()],
        Y=[qm.random_real_onb(), qm.random_real_onb()],
        c_onb=qm.random_real_onb()).reshape((2,) * 7).swapaxes(2, 3).reshape((128,))
    qm_cor_CO2 = one_switch_4mmts.qm_corr_one_switch_4mmts_3stngs(
        rho_ctb=qm.proj(qm.kron(qm.ket1, qm.ket0, qm.ket0)),
        X1=[qm.random_real_onb(), qm.random_real_onb()],
        X2=[qm.random_real_onb(), qm.random_real_onb()],
        Y=[qm.random_real_onb(), qm.random_real_onb()],
        c_onb=qm.random_real_onb()).reshape((2,) * 7).swapaxes(2, 3).reshape((128,))
    initial_guess = np.r_[[1], full_to_NSS_rep(qm_cor), full_to_NSS_rep(qm_cor_CO2)]
    print('Constructed qm_cor')
    sol = in_LC_scipy_root(qm_cor, np.random.rand(173))
