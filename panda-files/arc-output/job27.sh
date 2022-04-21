#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=4
#SBATCH --mem-per-cpu=8G
#SBATCH --time=120:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=tein.vanderlugt@wolfson.ox.ac.uk

module load scipy
module load sympy/1.6.2-foss-2020a-Python-3.8.2
time python3 towards_lc.py
