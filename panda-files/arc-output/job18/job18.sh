#!/bin/bash
#SBATCH --partition=short
#SBATCH --ntasks-per-node=2
#SBATCH --mem-per-cpu=8G
#SBATCH --time=12:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tein.vanderlugt@wolfson.ox.ac.uk

module load scipy
time python3 ../../../towards_lc.py
