import numpy as np
import utils

print("Reading...")
Q = utils.read_vertex_range_from_file('panda-files/results/8 all LC vertices', update_freq=1e5, batch_size=1e5).astype('int8')
print("\nSaving...")
np.save('panda-files/results/lc_vertices_int8.npy', Q)
print("Saved!")
