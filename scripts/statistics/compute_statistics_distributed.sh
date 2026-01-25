#!/bin/bash -x
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:A100:4
#SBATCH --cpus-per-task=28
#SBATCH --mem=120Gb
#SBATCH --time=04:00:00
#SBATCH --output=scripts/out/compute_statistics.%j
#SBATCH --job-name=compute_statistics

# Threading: avoid oversubscription
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR

EXPERIMENT="global"
MODALITY="seviri"

# (Optional but often helpful)
export NCCL_DEBUG=WARN
export PYTHONUNBUFFERED=1

# Use torchrun; SLURM provides a stable rendezvous endpoint (first node)
MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
MASTER_PORT=29500

srun ${WORKDIR}/.venv/bin/torchrun \
  --nnodes=1 \
  --nproc_per_node=4 \
  --rdzv_backend=c10d \
  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
  ${WORKDIR}/deep_lst/scripts/tools/compute_statistics.py \
    --modality $MODALITY \
    --statistics "assets/${MODALITY}/${EXPERIMENT}/statistics/statistics.npz" \
    --store "/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/${MODALITY}_test2.zarr" \
    --indices "assets/${MODALITY}/${EXPERIMENT}/indices/indices_subset.npz" \
    --batch_size 64 \
    --num_workers 20
