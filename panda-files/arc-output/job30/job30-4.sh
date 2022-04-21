#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=32G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tein.vanderlugt@wolfson.ox.ac.uk

time $DATA/cpanda -i safe -s lex_asc caus2_facets.pi
