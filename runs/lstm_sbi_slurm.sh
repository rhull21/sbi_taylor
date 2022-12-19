#!/bin/bash
#SBATCH --job-name=ensemble
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=4G
#SBATCH --time=12:00:00

module purge
module load parflow-ml
python lstm_sbi.py
