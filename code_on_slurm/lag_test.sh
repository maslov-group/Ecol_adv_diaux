#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --job-name=comm
#SBATCH --error=../all_slurm_out/comm_%A_%a.err
#SBATCH --output=../all_slurm_out/comm_%A_%a.out
#SBATCH --array=0-159
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

module purge
module load Python/3.10.1-IGB-gcc-8.2.0
# python lag_test.py $SLURM_ARRAY_TASK_ID
# python lag_test_vary_params.py $SLURM_ARRAY_TASK_ID
python lag_test_best_compare_nolag.py $SLURM_ARRAY_TASK_ID