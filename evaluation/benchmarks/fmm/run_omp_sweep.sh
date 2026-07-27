#!/bin/bash
#PBS -q regular-g
#PBS -l select=1
#PBS -l walltime=06:00:00
#PBS -W group_list=gc64
#PBS -j oe

cd "$PBS_O_WORKDIR"

for n in 50000000; do
  for theta in 0.2 0.25 0.3 0.35 0.4 0.5; do
    ./sweep_omp_dtt_openmp_cutoff.sh \
      --n "$n" --theta "$theta" \
      --depths "3,4,5,6,1000000" \
      --cutoffs "64,128,256,512,1024,2048,4096,8192,16384,32768" \
      --runs 10 \
      --out "sweep_logs/omp_cutoff_n${n}_theta${theta}.csv" \
      --raw-out "sweep_logs/omp_cutoff_n${n}_theta${theta}_raw.csv"
  done
done
