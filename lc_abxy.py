import itertools

import panda
import symmetry_utils
import utils
import vector_space_utils as vs
import numpy as np


def marginalise_vertices(input_file, output_file, use_numpy_load_save=False):
    # Instead, just marginalise c from the LC_acbxy vertices:
    marg_c_full_h = np.zeros((65, 129), dtype='int8')  # the marginalisation map \Sigma_c
    for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 0, b, x1, x2, y)] = 1
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 1, b, x1, x2, y)] = 1
    marg_c_full_h[-1, -1] = 1  # the homogeneous coordinates
    marg_c_nss_h = vs.construct_full_to_NSS_h(4, 2, 4, 2) @ marg_c_full_h @ vs.construct_NSS_to_full_h(8, 2, 4, 2)

    # lc_abxy_vertices = (marg_c_nss_h @ lc_vertices.T).T, which is:
    # lc_abxy_vertices = lc_vertices @ marg_c_nss_h.T

    if use_numpy_load_save:
        marginalised_vertices = np.load(input_file) @ marg_c_nss_h.T
        np.save(output_file, marginalised_vertices)
    else:
        marginalised_vertices = utils.read_vertex_range_from_file(input_file) @ marg_c_nss_h.T
        utils.write_rows_to_file(output_file, marginalised_vertices)


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
        for vertex in utils.read_vertex_range_from_file('panda-files/results/17 LC_abxy/lc_abxy_vertex_classes_with_duplicates'):
            f.write(panda.homog_vertex_to_str_with_fractions(vertex) + '\n')


def lc_abxy_facet_enum_panda_file(filename='panda-files/results/17 LC_abxy/4 lc_abxy_facet_enum.pi', readable=False):
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
    lines.append('Reduced Vertices:')
    for vertex in utils.read_vertex_range_from_file('panda-files/results/17 LC_abxy/3 lc_abxy_vertex_classes'):
        lines.append(panda.homog_vertex_to_str_with_fractions(vertex))

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
    return vs.construct_full_to_NSS_h(4, 2, 4, 2) @ perm_matrix @ vs.construct_NSS_to_full_h(4, 2, 4, 2)


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


def marginalise_ineqs(input_file, output_file):
    marg_c_full_h = np.zeros((65, 129), dtype='int8')  # the marginalisation map \Sigma_c
    for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 0, b, x1, x2, y)] = 1
        marg_c_full_h[vs.concatenate_bits(a1, a2, b, x1, x2, y)][vs.concatenate_bits(a1, a2, 1, b, x1, x2, y)] = 1
    marg_c_full_h[-1, -1] = 1  # the homogeneous coordinates
    marg_c_nss_h = vs.construct_full_to_NSS_h(4, 2, 4, 2) @ marg_c_full_h @ vs.construct_NSS_to_full_h(8, 2, 4, 2)

    new_ineqs = []

    for ineq in utils.read_vertex_range_from_file(input_file):
        # The following works:
        a_tilde = marg_c_nss_h @ ineq
        a_tilde[:12] = 1 / 2 * a_tilde[:12]
        a_tilde[14:-1] = 1 / 2 * a_tilde[14:-1]
        # This checks that it actually works, i.e. that the found inequality a_tilde is valid for LC_abxy: \Sigma_c^T a_tilde == a. (See [p228])
        assert np.all(marg_c_nss_h.T @ a_tilde == ineq)

        new_ineqs.append(a_tilde)

    utils.write_rows_to_file(output_file, new_ineqs)


def deterministic_cor_full(a1, a2, b):
    """ a1, a2, b should be functions (e.g. lambdas) from three binary variables to one binary variable. The returned vector is full and not homogeneous, so of length 64. """
    cor = np.zeros((2,) * 6, dtype='int')
    for _a1, _a2, _b, x1, x2, y in itertools.product((0, 1), repeat=6):
        cor[_a1, _a2, _b, x1, x2, y] = 1 * (_a1 == a1(x1, x2, y)) * (_a2 == a2(x1, x2, y)) * (_b == b(x1, x2, y))
    return cor.reshape((64,))


def facet1():
    ineq_full = np.zeros((2,) * 6)
    for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        if a1 == x2 and a2 == x1 and b == 1 and y == 1:
            ineq_full[a1, a2, b, x1, x2, y] += 1
        if b == 0 and y == 1:
            ineq_full[a1, a2, b, x1, x2, y] += 1 / 2
    ineq_full_h = np.r_[ineq_full.reshape(64), [-2]]
    ineq_nss_h = vs.construct_NSS_to_full_h(4, 2, 4, 2).T @ ineq_full_h
    return ineq_nss_h


def facet2try(alpha=.5):
    ineq_full = np.zeros((2,) * 6)
    for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
        if a1 == x2 and a2 == x1 and b == x1 and y == 1:
            ineq_full[a1, a2, b, x1, x2, y] += 1
        if b != x1 and y == 1:
            ineq_full[a1, a2, b, x1, x2, y] += alpha
    ineq_full_h = np.r_[ineq_full.flatten(), [0]]
    ineq_nss_h = vs.construct_NSS_to_full_h(4, 2, 4, 2).T @ ineq_full_h
    return ineq_nss_h


def maximum_violation_by_vertices(ineq, vertex_npy_file='panda-files/results/lc_abxy_vertices.npy'):
    # vertices = np.diagflat(np.reciprocal(vertices[:,-1])) @ vertices
    vertices = np.load(vertex_npy_file)
    violations = vertices @ ineq  # TODO normalise them
    argmax = np.argmax(violations)
    return violations[argmax], vertices[argmax]


def write_cor_to_file(cor_full, filename):
    with open(filename, 'w') as f:
        f.write('p(a1 a2 b | x1 x2 y)\n')
        for a1, a2, b, x1, x2, y in itertools.product((0, 1), repeat=6):
            i = vs.concatenate_bits(a1, a2, b, x1, x2, y)
            f.write('p(%d%d%d|%d%d%d): %s' % (a1, a2, b, x1, x2, y, str(cor_full[i])) + '\n')


if __name__ == '__main__':
    # marginalise_vertices(input_file='panda-files/results/7b lc_vertex_classes', output_file='panda-files/results/17 LC_abxy/lc_abxy_vertex_classes_with_duplicates')
    # reduce_vertices_panda_helper_file()
    # ** run panda-helper to reduce vertices (this also removes duplicates) **
    # lc_abxy_facet_enum_panda_file()
    # ** run panda on 'panda-files/results/17 LC_abxy/4 lc_abxy_facet_enum.pi' **
    # for known facets:
    # marginalise_ineqs('panda-files/results/12 facets adjacent to GYNI', 'panda-files/results/17 LC_abxy/known_facets_adjacent_to_GYNI')
    # marginalise_ineqs('panda-files/results/13 facets adjacent to LGYNI', 'panda-files/results/17 LC_abxy/known_facets_adjacent_to_LGYNI')

    ineq = facet2try().astype('int')
    print(' '.join(map(str, ineq)))
    violation, cor = maximum_violation_by_vertices(ineq)
    print(violation)
    write_cor_to_file((vs.construct_NSS_to_full_h(4,2,4,2) @ cor)[:-1], 'tmp')
