#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=01:00:00

module purge
module load parflow-ml
python storage.py