These files were used for research leading to the publication that can be found at https://arxiv.org/abs/2208.00719. 
Resulting data files can be found in [output](output). How these were generated is summarised in the notebook [main.ipynb](main.ipynb).

### LC vertices
Output file [output/4_LC_vertex_classes](output/4_LC_vertex_classes) contains the 219 vertex classes of the LCO polytope (here called LC). Each vertex is a correlation of the form $p(a_1 a_2 b c | x_1 x_2 y)$, where all seven variables are binary. In the file, the vertices are however represented as vectors of 87 integers. Because LC is a subset of the no-signalling polytope with respect to Bob being causally disjoint from Alice 1, Alice 2 and Charlie together, each correlation $p$ can be represented by Collins-Gisin coordinates with respect to that bipartition of the parties, viz. by the 86 parameters

$$
\{ p(a_1 a_2 c | x_1 x_2) \}_{a_1 a_2 c \neq 111} \cup \{ p(b=0 | y) \} \cup \{ p(a_1 a_2 c, b=0 | x_1 x_2 y) \}_{a_1 a_2 c \neq 111},
$$

where the setting variables $x_1, x_2, y$ always range over the binary values $\{0,1\}$ and each individual set of parameters between $\{  \}$ is ordered lexicographically.

The length-87 vectors found in [output/4_LC_vertex_classes](output/4_LC_vertex_classes) are homogeneous Collins-Gisin coordinates; that is, one obtains the above Collins-Gisin representation by dividing by the 87th integer.

The file [output/5_all_LC_vertices.npy](output/5_all_LC_vertices.npy) contains a numpy dump of an array of _all_ the LC vertices (not up to symmetries). This is a matrix of shape (9165312, 87), each row of which represents a vertex.

### LC1 vertices
The file [output/2_LC1_vertex_classes](output/2_LC1_vertex_classes) contains the vertex classes of the sub-polytope LC1. LC1 is subject to the additional no-signalling constraint that A2 cannot signal to A1 and B together and therefore has a lower dimension of 80. It admits a parameterisation in terms of the following 80 parameters:

$$
\{ p(a_1=0 | x_1) \} \cup \{ p(b=0 | y) \} \cup \{ p(a_1=0, b=0 | x_1 y) \} \cup \{ p(a_1 a_2 c | x_1 x_2) \}_{a_2 c \neq 11} \cup \{ p(a_1 a_2 c, b=0 | x_1 x_2 y) \}_{a_2 c \neq 11}.
$$

The file again contains homogeneous coordinates, so one should divide by the last entry of each row.

### Converting between representations
The Python file [vector_space_utils.py](vector_space_utils.py) contains functions that return matrices that convert between these different homogeneous coordinates: `construct_full_to_NSS_h`, `construct_NSS_to_full_h`, `construct_full_to_LC1_h` and `construct_LC1_to_full_h`. Here `NSS` refers to the Collins-Gisin representation discussed above ('no superluminal signalling' as opposed to the normal no-signalling, because we also have timelike no-signalling constraints in our scenario), and `h` denotes homogeneous coordinates.

Here, the 'full' representation is in terms of the probabilities $p(a_1 a_2 c b | x_1 x_2 y)$ directly (thus 128 parameters + a denominator). They are ordered lexicographically **with the important detail that throughout the code, c comes before b**: so b is the faster-changing variable in the lexicographic order. (This is to keep Alice and Charlie together because they belong on one side of the spacelike divide.)