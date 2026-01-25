#!/bin/bash -x
#SBATCH --partition=gpu # GPU queue
#SBATCH --gres=gpu:A100:1 # Syntax = gpu:<A40/A100>:<number of gpus>
#SBATCH --cpus-per-task=32 # one GPU : 32 threads (1/8 of the total available threads)
#SBATCH --mem=120GB # Memory per CPU, For A40: 120GB. for A100: 240GB
#SBATCH --nodes=1
#SBATCH --time=08:00:00
#SBATCH --output=scripts/out/train_resnet.%j
#SBATCH --job-name=train_resnet

# Parse command line argument for experiment name
CONFIGNAME=${1:-resnet}

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

srun python3 ${WORKDIR}/deep_lst/resnet/train.py --config ${CONFIGNAME}
