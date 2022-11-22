# This is all about the 86-dimensional instance of the LC polytope

import numpy as np

import panda
import quantum_utils as qm


def make_panda_file_with_closest_vertices(num_of_vertices, cor, filename):
    """ Make a PANDA input file for facet enumeration, containing the num_of_vertices vertices of the LC polytope
        closest to the point specified by cor. """
    all_vertices = np.load('panda-files/results/lc_vertices.npy')

    # Find closest vertices. First renormalise to a denominator of 2 to be able to calculate distance:
    assert np.max(all_vertices[:, -1]) <= 2
    for i in np.argwhere(all_vertices[:, -1] != 2):
        assert all_vertices[i, -1] == 1
        all_vertices[i] *= 2
    cor *= 2 / cor[-1]

    def get_closest_vertices_in_batch(batch):
        distances = np.linalg.norm(batch[:, :-1] - cor[:-1], axis=1)
        sorted_indices = np.argsort(distances)
        return batch[sorted_indices[:num_of_vertices]]

    batch_size = 100000
    i = 0
    # Find closest vertices in each batch:
    closest_vertices = np.empty((0, all_vertices.shape[1]), dtype='int')
    while i < len(all_vertices) + batch_size:
        print(f'i={i}', end='\r')
        closest_vertices = np.r_[closest_vertices, get_closest_vertices_in_batch(all_vertices[i:min(i + batch_size, len(all_vertices))])]
        i += batch_size
    del all_vertices
    # Finally, reduce this to the closest vertices overall
    closest_vertices = get_closest_vertices_in_batch(closest_vertices)

    # Write PANDA file
    with open(filename, 'w') as f:
        f.write(f'DIM={closest_vertices.shape[1] - 1}\n')
        f.write('Vertices:\n')
        for vertex in closest_vertices:
            f.write(panda.homog_vertex_to_str_with_fractions(vertex) + '\n')


if __name__ == '__main__':
    cor = qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_tcb_0phi,
                                    instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                    instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.z_onb, qm.ket1)],
                                    instr_C=qm.instr_vn_destr(qm.diag1_onb),
                                    instrs_B=[qm.instr_vn_destr(qm.z_onb), qm.instr_vn_destr(qm.diag1_onb)])
    # Try this one later:
    # cor = qm.quantum_cor_nss_noTmmt(rho_ctb=qm.rho_ctb_plusphiplus,
    #                                 instrs_A1=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
    #                                 instrs_A2=[qm.instr_measure_and_prepare(qm.z_onb, qm.ket0), qm.instr_measure_and_prepare(qm.x_onb, qm.ket1)],
    #                                 instr_C=qm.instr_vn_destr(qm.x_onb),
    #                                 instrs_B=[qm.instr_vn_destr(qm.diag1_onb), qm.instr_vn_destr(qm.diag2_onb)])

    make_panda_file_with_closest_vertices(500, cor, 'panda-files/cutting-polytope/0phi-c-diag1-b-zanddiag1')
    # TODO Problem: these vertices lie in a low-dimensional plane, so the inequalities I would get from this would not necessarily be facets of the original polytope.
