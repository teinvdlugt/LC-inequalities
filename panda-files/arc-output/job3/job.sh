#!/bin/bash
#SBATCH --partition=short
#SBATCH --ntasks-per-node=2

time ../panda -c -i safe nss_facets_4-5-2-2.pi
