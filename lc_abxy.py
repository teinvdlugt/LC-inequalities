import itertools

import panda
import symmetry_utils
import time
import utils
import vector_space_utils as vs
import numpy as np

# This is all unnecessary:
"""
def write_panda_input_for_lco1(filename='panda-files/lco1_abxy_facets.pi', readable=False):
    \""" See http://comopt.ifi.uni-heidelberg.de/software/PANDA/format.html for format of PANDA input file.
    :param filename: if None, then automatically generated.
    :param readable: whether the variable names should be human-readable. \"""
    lines = []

    # 1) Dimension information. This time, work in NSS coordinates all the way; don't use LCO1 coordinates
    dim_NSS = vs.dim_NSS(4, 2, 4, 2)  # 38
    lines.append('DIM=%d' % dim_NSS)

    # 2) Names of coordinates
    lines.append('Names:')
    if readable:
        var_names = panda.nss_readable_var_names(4, 2, 4, 2)
    else:
        var_names = ['x' + str(i) for i in range(0, dim_NSS)]
    lines.append(' '.join(var_names))

    # 3) Symmetry information
    lines.append('Maps:')
    for symm in lco1_symm_generators():
        lines.append(symmetry_utils.symm_matrix_to_string(symm, var_names))

    ## TODO the below isn't changed from acb_xy yet - I just realised that this is all unnecessary.

    # 5) Inequalities and equations
    lines.append('Inequalities:')
    # the ineqs are -p(a1a2cb|x1x2y) <= 0 for all a1,a2,c,b,x1,x2,y. For the final 8 rows, note that this means -(p(1111|x1x2y) - 1) - 1 <= 0.
    NtoF = vs.construct_NSCO1_to_full_weird_matrix().astype('int8')  # list of 'full vectors' expressed in NSCO1 vectors
    dim_full = 128
    for i in range(0, dim_full):
        lines.append(' '.join(map(str, -NtoF[i])) + (' 0' if i < 120 else ' -1'))

    # Write to file
    if filename is None:
        filename = 'panda-files/old-and-irrelevant/nsco1_facets_perm6feb.pi'
    file = open(filename, 'w')
    file.write('\n'.join(lines))
    file.close()


def full_perm_to_symm_h(perm, dtype='int'):
    return np.r_[
        np.array([
            utils.one_hot_vector(65, vs.concatenate_bits(*perm(*var_tuple)), dtype=dtype)
            for var_tuple in itertools.product((0, 1), repeat=6)
        ]),
        [utils.one_hot_vector(65, -1, dtype=dtype)]
    ]


def var_perm_to_symm_h(perm, dtype='int'):
    perm_matrix = full_perm_to_symm_h(perm, dtype)
    return vs.construct_full_to_NSS_homog(4, 2, 4, 2) @ perm_matrix @ vs.construct_NSS_to_full_homogeneous(4, 2, 4, 2)
"""


def construct_lc_abxy_vertices():
    # Instead, just marginalise c from the LC_acbxy vertices:
    marg_c_full_h = np.zeros((65, 129), dtype='int')  # the marginalisation map \Sigma_c
    for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 0, b, x1, x2, y)] = 1
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 1, b, x1, x2, y)] = 1
    marg_c_full_h[-1, -1] = 1  # the homogeneous coordinates
    marg_c_nss_h = vs.construct_full_to_NSS_homog(4, 2, 4, 2) @ marg_c_full_h @ vs.construct_NSS_to_full_homogeneous(8, 2, 4, 2)

    # Load LC_acbxy vertices
    # lc_vertices = utils.read_vertex_range_from_file('panda-files/results/8 all LC vertices', stop_at_excl=1000)

    # Make LC_abxy vertices
    # lc_abxy_vertices = (marg_c_nss_h @ lc_vertices.T).T, which is:
    # lc_abxy_vertices = lc_vertices @ marg_c_nss_h.T
    lc_abxy_vertices = np.load('panda-files/results/lc_vertices.npy') @ marg_c_nss_h.T

    np.save('panda-files/results/lc_abxy_vertices.npy', lc_abxy_vertices)


if __name__ == '__main__':
    construct_lc_abxy_vertices()
