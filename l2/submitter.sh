#!/bin/bash
#
#SBATCH --job-name=HJB_PINN_L2_TRAINING
#SBATCH --output=pinn_slurm_output.txt
#
#SBATCH --ntasks=1
#SBATCH --time=03-00:00:00
#SBATCH --mem=64GB

srun python -u HJB-PINN-L2.py > pinn_run_output.txt
