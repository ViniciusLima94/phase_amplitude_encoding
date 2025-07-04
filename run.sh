#!/bin/bash

#SBATCH -J proc              # Job name
#SBATCH -o log.out   # Name of stdout output file (%j expands to %jobID)
#SBATCH --ntasks=1
#SBATCH --mem-per-cpu=2000
#SBATCH --cpus-per-task=40

##SBATCH --array=0-25

python -O generate_hopf_dynamics.py
