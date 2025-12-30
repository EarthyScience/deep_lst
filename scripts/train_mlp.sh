#!/bin/bash -x

#SBATCH --partition=gpu # GPU queue
#SBATCH --gres=gpu:A100:4 # Syntax = gpu:<A40/A100>:<number of gpus>
#SBATCH --cpus-per-task=16 # one GPU : 32 threads (1/8 of the total available threads)
#SBATCH --mem=60G # Memory per CPU, For A40: 120GB. for A100: 240GB
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --time=24:00:00
#SBATCH --output=scripts/out/train_mlp.%j
#SBATCH --job-name=train_mlp

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst

cd $WORKDIR

set -euo pipefail


EXPERIMENT="baseline"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --experiment)   EXPERIMENT="$2"; shift 2 ;;
    --experiment=*) EXPERIMENT="${1#*=}"; shift ;;
  esac
done

source ${WORKDIR}/.venv/bin/activate
srun python3 ${WORKDIR}/deep_lst/mlp/trainer.py --experiment $EXPERIMENT