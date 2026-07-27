#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=02:00:00
#PBS -W group_list=gc64
#PBS -j oe

cd "$PBS_O_WORKDIR"
python tree_visualize_profile.py
