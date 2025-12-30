#!/bin/bash -x

#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=120Gb
#SBATCH --time=04:00:00
#SBATCH --output=scripts/out/compute_statistics.%j
#SBATCH --job-name=compute_statistics

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

WORKDIR=/Net/Groups/BGI/people/mwessel/deep_lst
cd $WORKDIR

EXPERIMENT="test"
MODALITY="seviri"

${WORKDIR}/.venv/bin/python3 ${WORKDIR}/deep_lst/compute_statistics.py \
    --modality $MODALITY \
    --statistics "experiments/${MODALITY}_statistics.parquet" \
    --store "/Net/Groups/BGI/work_5/scratch/mwessel/wegen-stores/${MODALITY}_test2.zarr" \
    --indices "/Net/Groups/BGI/people/mwessel/wedata/wedata/datasets/${MODALITY}/global_indices_subset.npz" \
    --device cuda:0 \
    --batch_size 2052 \
    --num_workers 24

