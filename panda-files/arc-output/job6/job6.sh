#!/bin/bash
#SBATCH --partition=devel
#SBATCH --ntasks-per-node=8

time ../panda -c -i safe nsco1_facets.pi --known-vertices=known_vertices_from_job4_reordered 2>&1 | tee out234.po
