#!/bin/bash -x
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --nodes=1
#SBATCH --time=12:00:00
#SBATCH --output=scripts/out/finetune.%j
#SBATCH --job-name=finetune

# Finetune a pre-trained MLP (Hydra overrides passed through to finetune.py).
# Submit (h24 run -> 3-day/72h horizon):
#   sbatch scripts/finetuning/run_finetune.sh horizon=h72 loss=smoothl1 from_run=v18_mlp_rollout_smoothl1_h24_full/version_0 tag=ft

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export NCCL_DEBUG=INFO
export TORCH_NCCL_BLOCKING_WAIT=1

cd /Net/Groups/BGI/people/mwessel/deep_lst || exit 1

uv run deep_lst/finetune.py "$@"
