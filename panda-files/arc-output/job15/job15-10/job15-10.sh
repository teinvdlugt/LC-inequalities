#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=256G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tein.vanderlugt@wolfson.ox.ac.uk

time $DATA/cpanda -c -i safe -s nz_desc ../lc_vertices.pi
