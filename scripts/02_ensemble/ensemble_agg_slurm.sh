#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --nodes=1
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=02:00:00

module purge
module load parflow-ml
python ensemble_agg.py