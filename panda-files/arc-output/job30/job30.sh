#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=1
#SBATCH --mem-per-cpu=32G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tein.vanderlugt@wolfson.ox.ac.uk

time $DATA/panda-data/panda -c -i safe -s reverse caus2_facets.pi
