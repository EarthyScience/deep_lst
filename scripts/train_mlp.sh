#!/bin/bash -x

#SBATCH --partition=gpu # GPU queue
#SBATCH --gres=gpu:A40:2 # Syntax = gpu:<A40/A100>:<number of gpus>
#SBATCH --cpus-per-task=32 # one GPU : 32 threads (1/8 of the total available threads)
#SBATCH --mem=120G # Memory per CPU, For A40: 120GB. for A100: 240GB
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --time=01:00:00
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

source ${WORKDIR}/.venv/bin/activate
srun python3 ${WORKDIR}/deep_lst/mlp/trainer.py