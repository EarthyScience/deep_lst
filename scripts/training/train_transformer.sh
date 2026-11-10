#!/bin/bash -x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=256G
#SBATCH --nodes=1
#SBATCH --time=28:00:00
#SBATCH --output=scripts/out/train_transformer.%j
#SBATCH --job-name=train_transformer

# Usage: sbatch train_transformer.sh [hydra_overrides...]
#
# Examples:
#   sbatch scripts/training/train_transformer.sh model=transformer objective=rollout loss=smoothl1 horizon=h24 data=full_glofas

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR
source ${WORKDIR}/.venv/bin/activate
export PYTHONPATH="$WORKDIR"

srun python3 ${WORKDIR}/deep_lst/train.py "$@"
