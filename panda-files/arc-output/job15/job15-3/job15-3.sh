#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00

time $DATA/panda-data/panda -c -i safe -s lex_asc ../lc_vertices.pi
