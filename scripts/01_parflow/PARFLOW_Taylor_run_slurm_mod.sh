#!/bin/bash
#SBATCH --job-name=ParFlow_Run
#SBATCH --nodes=1
#SBATCH --ntasks=9
#SBATCH --time=8:00:00
#SBATCH --gres=gpu:0


module purge
module load parflow-ml
python PARFLOW_Taylor_run.py
