import quantum_utils as qm
import numpy as np
import scipy
import face_utils as fc
import linprog as lp

cor = qm.quantum_cor_nss_discardT(rho_ctb=qm.rho_tcb_0phi,  # NOTE qm.rho_ctb_plusphiplus also works, if Alice does X mmt on setting x_i=1. Think about later
                                  instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                  instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                  # instr_C=qm.instr_vn_destr(qm.onb_from_direction(0.148 * np.pi)),  # 0.148 * np.pi seems to give maximal violation
                                  instr_C=qm.instr_vn_destr(qm.diag1_onb),
                                  instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.x_onb)])
print(lp.is_cor_in_LCstar(cor, tol=1e-8).success)  # NOTE: NOT IN LC* EITHER!

# Correlations that still require an explanation:
"""
cor = qm.quantum_cor_nss_discardT(rho_ctb=qm.rho_tcb_0phi,
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

"""cor = qm.quantum_cor_nss_discardT(rho_ctb=qm.rho_ctb_plusphiplus,
                                instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
                                instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
                                instr_C=qm.instr_vn_destr(qm.x_onb),
                                instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])
print(lp_is_cor_in_lc(cor).success)  # is outside LC
print_x12y_chsh_violations(cor)  # but doesn't violate a CHSH ineq btw b and c (neither for fixed x1 nor for fixed x2)
print(max_of_all_chsh_violations(cor))"""

## MAXIMISING OVER PROJECTIVE MEASUREMENTS
from math import cos, sin, pi

# This is the total LC value for the c = x2 a1 + (1-x2) c' strategy:
lc_value = lambda a, phi, b, chi, c, psi: 3 / 4 + 1 / 4 * cos(a) + 1 / 8 * (
        4 + sin(a) * sin(c) * cos(phi - psi) + cos(a) * cos(c) + sin(b) * sin(c) * cos(chi - psi) + cos(b) * cos(c) + cos(a) - cos(b))
x0 = np.array([0, 0, pi / 2, 0, pi / 4, 0])


def lc_value_array(x):
    return -lc_value(*x)


def lc_value_array_not_complex(x):
    return -lc_value(x[0], 0, x[1], 0, x[2], 0)


def lc_value_array_not_complex_cpiover4(x):
    return -lc_value(x[0], 0, x[1], 0, pi / 4, 0)


scipy.optimize.minimize(lc_value_array, x0=np.random.rand(6), method='powell')
scipy.optimize.minimize(lc_value_array, x0=np.random.rand(6), method='nelder-mead')
scipy.optimize.minimize(lc_value_array_not_complex, x0=np.zeros(3), method='powell')  # Yields the same value, so adding complex numbers doesn't yield any advantage here
scipy.optimize.minimize(lc_value_array_not_complex_cpiover4, x0=np.zeros(2), method='powell')  # Yields the result that Wolfram gave me (maximum when c=pi/4 is fixed)

max = 0
argmax = []
for _ in range(200):
    result = scipy.optimize.minimize(lc_value_array_not_complex, x0=np.random.rand(3) * 2 * pi, method='powell')
    if -result.fun > max:
        max = -result.fun
        argmax = result.x
print(max)
lc_value(argmax[0] % pi, 0, argmax[1], 0, argmax[2], 0)
print(argmax)
print((argmax + pi) % (2 * pi) - pi)
# Cosistently yields   argmax = +/-[0.27564432, 2.18628108, 1.23096724]  yielding LC value of 1.827350269

# found_maxima = np.array(found_maxima)
# plt.hist(x=found_maxima, bins='auto')
# plt.show()
# print(max(found_maxima))

# found_maxima = []
# for _ in range(1000):
#     result = scipy.optimize.minimize(lc_value_array, x0=np.random.rand(6) * 2 * pi, method='powell')
#     found_maxima.append(-result.fun)
# found_maxima = np.array(found_maxima)
# plt.hist(x=found_maxima, bins='auto')
# plt.show()
# print(max(found_maxima))

## New inequalities in light of possibilistic lemma
ineq_p14 = fc.construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == a1 and x2 == 1 and y == 0) * 1 / 2
                                                                 - (a2 == 1 and x1 == 0 and x2 == 1) * 1 / 2
                                                                 - (a1 == 1 and x1 == 1 and x2 == 0) * 1 / 2
                                                                 - (a1 == a2 == 0 and x1 == x2 == 1) * 1 / 2
                                                                 + ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)  # dim 40
ineq_p14_lazier = fc.construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: (b == a1 and x2 == 1 and y == 0 and x1 == 1)
                                                                        - (a2 == 1 and x1 == 0 and x2 == 1 and y == 0)
                                                                        - (a1 == 1 and x1 == 1 and x2 == 0 and y == 0)
                                                                        - (a1 == a2 == 0 and x1 == x2 == 1 and y == 0)
                                                                        + ((b + c) % 2 == x1 * y and x2 == 0) * 1 / 4, 7 / 4)  # dim 41
cor2 = qm.quantum_cor_nss_discardT(rho_ctb=qm.rho_tcb_0phi,
                                   instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                   instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                   # instr_C=qm.instr_vn_destr(qm.onb_from_direction(0.148 * np.pi)),  # 0.148 * np.pi seems to give maximal violation
                                   instr_C=qm.instr_vn_destr(qm.x_onb),
                                   instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])
ineq_p14_2 = fc.construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: - (a2 == 1 and x1 == 0) * 1 / 4
                                                                   - (a1 == 1 and x2 == 0) * 1 / 4
                                                                   # - (a1 == a2 == 0 and x1 == x2 == 1) * 1 / 2
                                                                   + ((x1 * a1 + (1 - x1) * c + b) % 2 == x1 * y and x2 == x1) * 1 / 4, 3 / 4)  # dim 49
ineq_p14_2_lazier = fc.construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: - (a2 == 1 and x1 == 0 and x2 == 1 and y == 0)
                                                                          - (a1 == 1 and x1 == 1 and x2 == 0 and y == 0)
                                                                          #       - (a1 == a2 == 0 and x1 == x2 == 1 and y == 0)
                                                                          + ((x1 * a1 + (1 - x1) * c + b) % 2 == x1 * y and x2 == x1) * 1 / 4, 3 / 4)  # dim 49
ineq_p19 = fc.construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: - (a1 == 1 and x1 == 1 and x2 == 0) * 1 / 4
                                                                 - (a2 == 1 and x1 == 0 and x2 == 1) * 1 / 4
                                                                 - (a1 == a2 == 0 and x1 == x2 == 1) * 1 / 4
                                                                 + ((x1 * a1 + (1 - x1) * c + b) % 2 == x1 * y and x2 == x1) * 1 / 4, 3 / 4)  # dim 61
ineq_p19_lazier = fc.construct_ineq_nss(7, lambda a1, a2, c, b, x1, x2, y: - (a1 == 1 and x1 == 1 and x2 == 0 and y == 0) * 1 / 2
                                                                        - (a2 == 1 and x1 == 0 and x2 == 1 and y == 0) * 1 / 2
                                                                        - (a1 == a2 == 0 and x1 == x2 == 1 and y == 0) * 1 / 2
                                                                        + ((x1 * a1 + (1 - x1) * c + b) % 2 == x1 * y and x2 == x1) * 1 / 4, 3 / 4)  # dim 61
print(fc.max_violation_by_lc(ineq_p19_lazier)[0])
print(cor2 @ ineq_p19_lazier)
