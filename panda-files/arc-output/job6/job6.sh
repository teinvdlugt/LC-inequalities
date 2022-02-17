#!/bin/bash
#SBATCH --partition=devel
#SBATCH --ntasks-per-node=1

time $DATA/panda-data/panda -c -i safe nsco1_facets.pi --known-vertices=known_vertices_from_job4_reordered
