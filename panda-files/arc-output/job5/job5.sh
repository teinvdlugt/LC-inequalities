#!/bin/bash
#SBATCH --partition=long
#SBATCH --ntasks-per-node=28
#SBATCH --time=120:00:00

time ../panda -c -i safe -s reverse nsco1_facets.pi
