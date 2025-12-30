#!/bin/bash -x

#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128Gb
#SBATCH --time=01:30:00
#SBATCH --output=scripts/out/inference.%j
#SBATCH --job-name=inference

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
srun python3 ${WORKDIR}/deep_lst/mlp/inference.py --experiment $EXPERIMENT