#!/bin/bash
#SBATCH --job-name=ParFlow_Run
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --time=12:00:00


module purge
module load parflow-ml
python PARFLOW_Taylor_run.py
