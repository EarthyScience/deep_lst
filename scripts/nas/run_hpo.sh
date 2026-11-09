#!/bin/bash -x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --time=4-00:00:00
#SBATCH --output=scripts/out/hpo.%j
#SBATCH --job-name=hpo

# Usage:
#   sbatch scripts/nas/run_hpo.sh                      # 50 trials
#   HPO_N_TRIALS=50 sbatch scripts/nas/run_hpo.sh     # custom trial count
#
# Parallel workers sharing the same SQLite study:
#   sbatch scripts/nas/run_hpo.sh; sbatch nas/run_hpo.sh; sbatch nas/run_hpo.sh

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

python3 ${WORKDIR}/deep_lst/nas/hpo.py
