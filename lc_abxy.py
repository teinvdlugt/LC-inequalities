import itertools

import panda
import symmetry_utils
import time
import utils
import vector_space_utils as vs
import numpy as np


def construct_lc_abxy_vertices():
    # Instead, just marginalise c from the LC_acbxy vertices:
    marg_c_full_h = np.zeros((65, 129), dtype='int')  # the marginalisation map \Sigma_c
    for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 0, b, x1, x2, y)] = 1
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 1, b, x1, x2, y)] = 1
    marg_c_full_h[-1, -1] = 1  # the homogeneous coordinates
    marg_c_nss_h = vs.construct_full_to_NSS_homog(4, 2, 4, 2) @ marg_c_full_h @ vs.construct_NSS_to_full_homogeneous(8, 2, 4, 2)

    # Load LC_acbxy vertices
    lc_vertices = utils.read_vertex_range_from_file('panda-files/results/8 all LC vertices', start_at_incl=50000, stop_at_excl=51000)
    np.save('panda-files/results/lc_vertices.npy', lc_vertices)

    # Make LC_abxy vertices
    # lc_abxy_vertices = (marg_c_nss_h @ lc_vertices.T).T, which is:
    # lc_abxy_vertices = lc_vertices @ marg_c_nss_h.T
    lc_abxy_vertices = np.load('panda-files/results/lc_vertices.npy') @ marg_c_nss_h.T

    np.save('panda-files/results/lc_abxy_vertices.npy', lc_abxy_vertices.astype('int8'))


def reduce_vertices_panda_helper_file(filename='panda-files/lc_abxy_reduce_vertices'):
    with open(filename, 'w') as f:
        f.write('DIM=38\n')
        f.write('Names:\n')
        var_names = ['x' + str(i) for i in range(0, 38)]
        f.write(' '.join(var_names) + '\n')

        # Maps
        f.write('Maps:\n')
        for symm in lco1_symm_generators():
            f.write(symmetry_utils.symm_matrix_to_string(symm, var_names) + '\n')

        # Vertices to reduce
        f.write('Vertices:\n')
        for vertex in np.load('panda-files/results/lc_abxy_vertices.npy'):
            f.write(panda.row_with_denom_to_vector_str(vertex) + '\n')


def lc_abxy_facet_enum_panda_file(filename='panda-files/lc_abxy_vertices.pi', readable=False):
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

    # 5) Vertices
    lines.append('Reduced vertices:')
    # TODO load reduced vertices here. First reduce vertices.

    # Write to file
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


def lco1_symm_generators():
    return np.array([
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: (a1, a2, b, (x1 + 1) % 2, x2, y)),  # x1 -> x1 + 1
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: ((a1 + x1) % 2, a2, b, x1, x2, y)),  # a1 -> a1 + x1
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: (a1, a2, b, x1, (x2 + 1) % 2, y)),  # x2 -> x2 + 1
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: (a1, (a2 + x2) % 2, b, x1, x2, y)),  # a2 -> a2 + x2
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: (a1, a2, b, x1, x2, (y + 1) % 2)),  # y  -> y  + 1
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: (a1, a2, (b + y) % 2, x1, x2, y)),  # b  -> b  + y
        var_perm_to_symm_h(lambda a1, a2, b, x1, x2, y: (a2, a1, b, x2, x1, y))  # a2 <-> a1, x2 <-> x1
    ])


if __name__ == '__main__':
    construct_lc_abxy_vertices()
    reduce_vertices_panda_helper_file()
