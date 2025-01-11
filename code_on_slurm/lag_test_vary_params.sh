#!/bin/bash
#SBATCH -p normal
#SBATCH -N 1
#SBATCH --job-name=meta_strat_lag_test_vary_params
#SBATCH --error=../all_slurm_out/meta_strat_lag_test_vary_params_%A_%a.err
#SBATCH --output=../all_slurm_out/meta_strat_lag_test_vary_params_%A_%a.out
#SBATCH --array=0-279
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1G

module purge
module load Python/3.10.1-IGB-gcc-8.2.0
python lag_test_vary_params.py $SLURM_ARRAY_TASK_ID