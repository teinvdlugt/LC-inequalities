#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=120:00:00

time $DATA/panda-data/panda -c -i safe -s reverse nsco1_facets_NSS_coords_H_symms.pi
