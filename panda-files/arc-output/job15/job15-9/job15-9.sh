#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=128G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tein.vanderlugt@wolfson.ox.ac.uk

time $DATA/cpanda -c -i safe ../lc_vertices_shuffled.pi
