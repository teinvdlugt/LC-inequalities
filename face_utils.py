import numpy as np
import itertools

import scipy
import sympy

import vector_space_utils as vs
import linprog as lp


def construct_ineq_full(num_of_binary_vars, point_function, upper_bound=0.0):
    """ point_function: A function of num_of_binary_vars binary variables, which returns the points won by the parties when they output those binary variables. """
    ineq_full = np.zeros((2,) * num_of_binary_vars)
    for var_tuple in itertools.product((0, 1), repeat=num_of_binary_vars):
        ineq_full[var_tuple] += point_function(*var_tuple)
    ineq_full_h = np.r_[ineq_full.flatten(), [-upper_bound]]
    return ineq_full_h


def construct_ineq_nss(point_function, upper_bound=0.0, num_of_binary_vars=7, na=8, nb=2, nx=4, ny=2):
    ineq_nss_h = vs.construct_NSS_to_full_h(na, nb, nx, ny).T @ construct_ineq_full(num_of_binary_vars, point_function, upper_bound)
    return ineq_nss_h


def max_violation_by_lc(ineq_nss):
    return lp.max_violation_by_LC(ineq_nss)[0]


def count_face_dimension(ineq, polytope_vertices_npy='output/5_all_LC_vertices.npy'):
    print('Computing violations...', end='\r')
    vertices = np.load(polytope_vertices_npy)
    violations = ineq @ vertices.T  # NOTE these violations might not be normalised. But for now only positivity/negativity matters

    # Check whether inequality is valid and tight for polytope
    max_violation = np.max(violations)
    if max_violation > 0:
        print("The inequality        ")
        print(' '.join(map(str, ineq)))
        print("is NOT valid for LC, as it is violated by %s (unnormalised) by the LC vertex" % str(max_violation))
        print(' '.join(map(str, vertices[np.argmax(violations)[0]])))
        return -1
    elif max_violation < 0:
        print("The inequality is satisfied by the polytope but is not a strict inequality and hence does not support a face.")
        return 0

    # Count number of affinely independent vertices on the face
    print("Computing dimension...", end='\r')
    vertices = vertices[np.argwhere(violations == 0).flatten()]
    dim = np.linalg.matrix_rank(vertices) - 1
    print("The inequality supports a face of dimension %d" % dim)
    return dim


def find_maximal_lin_indep_subset(matrix, lin_indep_subset=None, max_dimension=None, batch_size=1000):
    """
    Returns a maximal set of linearly independent rows from the given matrix.
    :param matrix: numpy matrix
    :param lin_indep_subset: start from this subset. The returned subset will be a superset of this
    :param max_dimension: algorithm stops when this number of lin indep rows is found
    :param batch_size: affects efficiency of the algorithm
    """
    if lin_indep_subset is None:
        lin_indep_subset = np.empty((0, matrix.shape[1]), matrix.dtype)
    if max_dimension is None or max_dimension > matrix.shape[1]:
        max_dimension = matrix.shape[1]

    for i in range(0, len(matrix), batch_size):
        batch = matrix[i: min(i + batch_size, len(matrix))]
        # Check if batch contains any vertices that are lin indep of lin_indep_subset
        num_of_new_lin_indep_vectors = np.linalg.matrix_rank(np.r_[lin_indep_subset, batch]) - len(lin_indep_subset)
        if num_of_new_lin_indep_vectors > 0:
            # Check all vertices for affine independence individually
            new_aff_indep_vertices_found = 0
            for v in batch:
                maybe_lin_indep_subset = np.r_[lin_indep_subset, [v]]
                if np.linalg.matrix_rank(maybe_lin_indep_subset) == len(maybe_lin_indep_subset):
                    # Found affinely independent vertex
                    lin_indep_subset = maybe_lin_indep_subset

                    if len(lin_indep_subset) >= max_dimension:
                        return lin_indep_subset

                    new_aff_indep_vertices_found += 1
                    if new_aff_indep_vertices_found == num_of_new_lin_indep_vectors:
                        break
        print('Finding aff indep vertices on face... %d / %d' % (i, len(matrix)), end='\r')  # this message makes sense when used in find_a_facet_adjacent_to_face
    return lin_indep_subset


def find_a_facet_adjacent_to_face(face, vertices, aff_indep_vertices_on_face=None, tol=1e-13, randomise=True):
    d = vertices.shape[1] - 1

    print("Computing violations...", end='\r')
    violations = vertices @ face
    if np.max(violations) > tol:
        print("\nThis is not a face!")
        print(np.max(violations))
        return -1

    if aff_indep_vertices_on_face is None:
        aff_indep_vertices_on_face = np.empty((0, d + 1), vertices.dtype)
    else:
        assert np.linalg.matrix_rank(aff_indep_vertices_on_face) == len(aff_indep_vertices_on_face)
        if len(aff_indep_vertices_on_face) >= d:
            return face

    # Find maximal set of affinely independent vertices on the face
    all_vertices_on_face = vertices[np.argwhere(np.abs(violations) < tol).flatten()]
    aff_indep_vertices_on_face = find_maximal_lin_indep_subset(all_vertices_on_face, aff_indep_vertices_on_face, max_dimension=d)
    if len(aff_indep_vertices_on_face) == d:
        print('\nFound facet!')
        return face
    # Dimension of the face:
    f = len(aff_indep_vertices_on_face) - 1
    print('Finding aff indep vertices on face... Done, f=%d            ' % f)

    # Throw away vertices that are on the face
    vertices = vertices[violations < -tol]

    # Need a d-2 dimensional rotation axis intersecting the face, spanned by d-1 affinely independent vectors
    # So need to find  (d-1)-(f+1)=d-f-2  remaining aff indep vectors on the face.
    # I.e. these vectors should be orthogonal to the normal to the face, and to all aff_indep_vertices_on_face
    def find_affinely_independent_points_on_hyperplane(points_h, hyperplane):
        x_0 = points_h[0][:-1]
        lambda_0 = points_h[0][-1]
        if len(points_h) == 1:
            pts_translated = np.empty((0, len(points_h[0]) - 1))
        else:
            pts_translated = np.array([pt[:-1] * lambda_0 / pt[-1] - x_0 for pt in points_h[1:]])
        indep_pts_translated = scipy.linalg.null_space(np.r_[[hyperplane[:-1]], pts_translated]).T  # TODO replace with integer method? (avoid normalisation)
        indep_pts = np.array([1 / lambda_0 * (pt + x_0) for pt in indep_pts_translated])  # TODO could improve randomisation by applying a rotation to indep_pts here
        indep_pts_h = np.concatenate([indep_pts, np.ones((len(indep_pts), 1), dtype=indep_pts.dtype)], axis=1)  # TODO try to normalise
        return indep_pts_h
    aff_indep_pts = find_affinely_independent_points_on_hyperplane(aff_indep_vertices_on_face, face)
    assert np.linalg.matrix_rank(np.r_[aff_indep_vertices_on_face, aff_indep_pts]) == len(aff_indep_vertices_on_face) + len(aff_indep_pts)
    assert len(aff_indep_pts) == d - f - 1
    m_index = np.random.randint(len(aff_indep_pts)) if randomise else len(aff_indep_pts) - 1
    remaining_axis_points = np.r_[aff_indep_pts[:m_index], aff_indep_pts[m_index + 1:]]
    assert len(remaining_axis_points) == d - f - 2

    axis_points = np.r_[aff_indep_vertices_on_face, remaining_axis_points]
    vertex_candidate = vertices[0]
    face_candidate = scipy.linalg.null_space(np.r_[axis_points, [vertex_candidate]]).flatten()
    batch_size = 10
    i = 1
    while i < len(vertices):
        if i % 1000 == 1:
            print('Rotating face... f=%d, %d / %d' % (f, i, len(vertices)), end='\r')
        batch = vertices[i: min(i + batch_size, len(vertices))]
        max_violating_vertex = batch[np.argmax(face_candidate @ batch.T)]
        # NOTE max_violating_vertex might actually not literally be the vertex with maximum violation, because we haven't normalised (i.e. divided by the homogeneous coordinate). but doesn't matter
        if face_candidate @ max_violating_vertex > tol:
            face_candidate = scipy.linalg.null_space(np.r_[axis_points, [max_violating_vertex]]).flatten()
            # Make sure that the previous vertex candidate lies on the good side of the new face: if not, flip the new face's sign
            if face_candidate @ vertex_candidate > -tol:
                face_candidate = -face_candidate
            vertex_candidate = max_violating_vertex
            # Redo this batch, so don't increment i.
        else:
            i += batch_size

    # Old, slower code:
    """
    axis_points = np.r_[aff_indep_vertices_on_face, remaining_axis_points]
    vertex_candidate = vertices[0]  # (NB this is not on the face)
    face_candidate = scipy.linalg.null_space(np.r_[axis_points, [vertex_candidate]]).flatten()
    for i in range(1, len(vertices)):
        if i % 1000 == 0:
            print('Rotating face... f=%d, i=%d' % (f, i), end='\r')
        v = vertices[i]
        if face_candidate @ v > 0:  # TODO maybe make this more efficient by calculating violations in batches
            face_candidate = scipy.linalg.null_space(np.r_[axis_points, [v]]).flatten()
            # Make sure that the previous vertex candidate lies on the good side of the new face: if not, flip the new face's sign
            if face_candidate @ vertex_candidate > -tol:
                face_candidate = -face_candidate
            vertex_candidate = v
    """

    print()
    # face_candidate should now be a new face!
    new_face = face_candidate
    new_vertex_on_face = vertex_candidate
    # print('Verifying new face...')
    # assert np.all(vertices @ new_face < tol)
    # print('New face: %s' % str(new_face))

    aff_indep_vertices_on_new_face = np.r_[aff_indep_vertices_on_face, [new_vertex_on_face]]
    assert np.linalg.matrix_rank(aff_indep_vertices_on_new_face) == len(aff_indep_vertices_on_new_face)

    if len(aff_indep_vertices_on_new_face) == d:
        print("Computing exact result...", end='\r')
        facet_exact = np.array(sympy.Matrix(aff_indep_vertices_on_new_face).nullspace()[0][:], dtype='int')  # i.e. without numerical errors
        print("Verifying result...      ")
        violations = vertices @ facet_exact
        if np.max(violations) > 0:
            assert np.min(violations) == 0
            facet_exact = -facet_exact
        print('Found facet with dimension %d!' % (len(aff_indep_vertices_on_new_face) - 1))
        return facet_exact
    else:
        print("Found new face with dimension", len(aff_indep_vertices_on_new_face) - 1)
        return find_a_facet_adjacent_to_face(new_face, vertices, aff_indep_vertices_on_new_face, tol, randomise)


# NOTE not sure if I finished this:
def find_facet_violated_by_point(point, vertices):
    d = vertices.shape[1] - 1

    # Sort vertices based on distance to point. Closer-by vertices are more likely to be on a facet violated by the point.
    """print('Sorting vertices by distance...')
    assert np.max(vertices[:, -1]) <= 2
    for i in np.argwhere(vertices[:, -1] != 2):
        assert vertices[i, -1] == 1
        vertices[i] *= 2
    point = 2 / point[-1] * point
    batch_size = 10000
    distances = np.empty((0,), dtype='float')
    # continue optimising?
    distances = np.linalg.norm(vertices[:, :-1] - point[:-1], axis=1)
    print('Really sorting...')
    sorted_indices = np.argsort(distances)
    print('Done really sorting')
    vertices = vertices[sorted_indices]"""

    # Start with an informed guess: pick hyperplane through closest affinely independent vertices
    print('Finding aff indep vertices...')
    aff_indep_vertices = np.empty((0, d + 1), dtype='int8')
    i = 0
    while i < len(vertices):
        new_matrix = np.r_[aff_indep_vertices, [vertices[i]]]
        if np.linalg.matrix_rank(new_matrix) == len(new_matrix):
            aff_indep_vertices = new_matrix
            if len(aff_indep_vertices) == d:
                i += 1
                break
        i += 1
    facet_candidate = scipy.linalg.null_space(aff_indep_vertices).flatten()
    # facet_candidate = np.array(sympy.Matrix(aff_indep_vertices).nullspace()[0][:], dtype='int')
    assert len(facet_candidate) == d + 1

    # Get the point on the right side of the inequality
    facet_candidate *= int(np.sign(facet_candidate @ point)) or 1  # np.sign(0)=0, but we want 1 in that case

    # Now pick the closest-by vertex that violates the inequality, and let it replace one of the aff_indep_vertices, chosen (randomly, for now) such that
    # the thrown-away vertex and the point lie on different sides of the new hyperplane.
    # continue where we were with i
    candidate_count = 0
    while i < len(vertices):
        if facet_candidate @ vertices[i] > 1e-15:  # due to ridiculous error size in scipy.linalg.null_space
            # Find vertex 'discard' in aff_indep_vertices s.t. hyperplane through aff_indep_vertices['discard'] + [vertices[i]] separates 'discard' and 'point'
            candidate_count += 1
            found_a_new_candidate = False
            for j in range(d - 1, -1, -1):
                new_aff_indep_vertices = aff_indep_vertices.copy()
                new_aff_indep_vertices[j] = vertices[i]
                new_facet_candidate = find_vector_in_kernel(new_aff_indep_vertices)
                # new_facet_candidate = np.array(sympy.Matrix(new_aff_indep_vertices).nullspace()[0][:], dtype='int')
                if len(new_facet_candidate) != d + 1:
                    print('Problem! %d' % len(new_facet_candidate))
                if (new_facet_candidate @ aff_indep_vertices[j]) * (new_facet_candidate @ point) < 0:
                    # Success
                    facet_candidate = new_facet_candidate * int(np.sign(new_facet_candidate @ point) or 1)
                    aff_indep_vertices = new_aff_indep_vertices
                    found_a_new_candidate = True
                    break
            if not found_a_new_candidate:
                print('THERE WAS NO VERTEX THAT COULD BE THROWN AWAY! this shouldn\'t happen')
        print('At i=%d, candidates tried: %d' % (i, candidate_count), end='\r')
        i += 1

    # facet_candidate should now be really a facet
    print('Validating facet...')
    max_violation_by_lc = np.max(vertices @ facet_candidate)
    if max_violation_by_lc > 0:
        print('\n\nError: LC violates the supposed facet by %s\n\n' % str(max_violation_by_lc))
    print('point violates facet by %s' % str(point @ facet_candidate))
    print('Facet is:')
    print(facet_candidate)

    return facet_candidate

def find_vector_in_kernel(matrix):
    # Use ILP for this. Probably Gaussian elimination would be faster, but I don't have time to write that myself
    # import pulp
    # model = pulp.LpProblem()
    # vars = []
    # for i in range(matrix.shape[1]):
    #     vars.append(pulp.LpVariable(name='x%d' % i, cat='Integer'))
    # model += ()

    # return np.linalg.lstsq(matrix, np.zeros(len(matrix), dtype='int'))[-1]

    from sympy import Matrix
    from diophantine import solve
    return np.array(i for i in solve(Matrix(matrix), Matrix(np.zeros(matrix.shape[0], dtype='int'))))