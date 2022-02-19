#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=16G
#SBATCH --time=120:00:00

time $DATA/panda-data/panda -c -i safe -s reverse nsco1_facets.pi --known-vertices=known_vertices_from_job4_reordered
