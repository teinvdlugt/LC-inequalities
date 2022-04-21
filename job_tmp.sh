#!/bin/bash
#SBATCH --partition=short
#SBATCH --time=12:00:00
#SBATCH --ntasks-per-node=16
#SBATCH --mem-per-cpu=2G

module load scipy
module load sympy/1.6.2-foss-2020a-Python-3.8.2
time python3 quantum_utils.py
