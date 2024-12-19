#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --job-name=phi_pre
#SBATCH --error=../all_slurm_out/phi_pre_%A_%a.err
#SBATCH --output=../all_slurm_out/phi_pre_%A_%a.out
#SBATCH --array=0-299
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

module purge
module load Python/3.10.1-IGB-gcc-8.2.0
python phi_pre.py $SLURM_ARRAY_TASK_ID